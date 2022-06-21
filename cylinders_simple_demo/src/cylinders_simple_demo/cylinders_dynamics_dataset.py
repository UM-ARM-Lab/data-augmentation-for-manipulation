import pathlib
import pickle
from functools import lru_cache
from multiprocessing import get_context
from typing import List, Optional, Callable, OrderedDict, Dict

import numpy as np
from torch.utils.data import Dataset

from cylinders_simple_demo.cylinders_scenario import CylindersScenario
from cylinders_simple_demo.data_utils import load_gzipped_pickle
from cylinders_simple_demo.utils import load_params, load_hjson
from link_bot_data.dataset_utils import batch_sequence
from moonshine.indexing import index_time_batched
from moonshine.numpify import numpify
from moonshine.tensorflow_utils import batch_examples_dicts
from moonshine.torch_and_tf_utils import remove_batch


class EmptyDatasetException(Exception):
    pass


def load_metadata(metadata_filename):
    if 'hjson' in metadata_filename.name:
        metadata = load_hjson(metadata_filename)
    elif 'pkl' in metadata_filename.name:
        with metadata_filename.open("rb") as f:
            metadata = pickle.load(f)
    else:
        raise NotImplementedError()
    metadata['filename'] = metadata_filename.stem
    metadata['example_idx'] = int(metadata_filename.stem[8:])
    metadata['full_filename'] = metadata_filename.as_posix()
    return metadata


@lru_cache
def load_single(metadata_filename: pathlib.Path, no_update_with_metadata=False):
    metadata = load_metadata(metadata_filename)

    data_filename = metadata.pop("data")
    full_data_filename = metadata_filename.parent / data_filename
    if str(data_filename).endswith('.gz'):
        example = load_gzipped_pickle(full_data_filename)
    else:
        with full_data_filename.open("rb") as f:
            example = pickle.load(f)
    example['metadata'] = metadata
    if not no_update_with_metadata:
        example.update(metadata)
    return example


def load_mode_filenames(d: pathlib.Path, filenames_filename: pathlib.Path):
    with filenames_filename.open("r") as filenames_file:
        filenames = [l.strip("\n") for l in filenames_file.readlines()]
    return [d / p for p in filenames]


def get_filenames(d, mode: str):
    all_filenames = []
    if mode == 'all':
        all_filenames.extend(load_mode_filenames(d, d / f'train.txt'))
        all_filenames.extend(load_mode_filenames(d, d / f'test.txt'))
        all_filenames.extend(load_mode_filenames(d, d / f'val.txt'))
    elif mode == 'notrain':
        all_filenames.extend(load_mode_filenames(d, d / f'test.txt'))
        all_filenames.extend(load_mode_filenames(d, d / f'val.txt'))
    elif mode == 'notest':
        all_filenames.extend(load_mode_filenames(d, d / f'train.txt'))
        all_filenames.extend(load_mode_filenames(d, d / f'val.txt'))
    else:
        filenames_filename = d / f'{mode}.txt'
        all_filenames.extend(load_mode_filenames(d, filenames_filename))
    all_filenames = sorted(all_filenames)
    return all_filenames


def pprint_example(example):
    for k, v in example.items():
        if hasattr(v, 'shape'):
            print(k, v.shape, v.dtype)
        elif isinstance(v, OrderedDict):
            print(k, numpify(v))
        else:
            print(k, type(v))


def process_filenames(filenames, process_funcs):
    filenames_list = filenames
    for p in process_funcs:
        filenames_list = p(filenames_list)
    return filenames_list


def default_get_filenames(filenames):
    for filename in filenames:
        yield filename


class NewBaseDataset:

    def __init__(self,
                 loader,
                 filenames: List,
                 mode,
                 n_prefetch=2,
                 post_process: Optional[List[Callable]] = None,
                 process_filenames: Optional[List[Callable]] = None,
                 ):
        self.loader = loader
        self.mode = mode
        self.filenames = filenames
        self.n_prefetch = n_prefetch
        if post_process is None:
            self._post_process = []
        else:
            self._post_process = post_process

        if process_filenames is None:
            self._process_filenames = []
        else:
            self._process_filenames = process_filenames

    def __iter__(self):
        generator = self.iter_multiprocessing()

        for example in generator:
            # NOTE: This post_process with both batched/non-batched inputs which is annoying
            example = self.loader.post_process(example)
            for p in self._post_process:
                example = p(example)

            yield example

    def iter_multiprocessing(self):
        assert self.n_prefetch > 0

        for idx, filenames_i in enumerate(process_filenames(self.filenames, self._process_filenames)):
            if isinstance(filenames_i, list):
                examples_i = list(self.loader.pool.map(load_single, filenames_i))
                example = batch_examples_dicts(examples_i)
            else:
                example = load_single(filenames_i)
            yield example

    def shuffle(self, seed: Optional[int] = 0, reshuffle_each_iteration=True):
        if not reshuffle_each_iteration:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()

        def _shuffle_filenames(filenames):
            shuffled_filenames = filenames.copy()
            rng.shuffle(shuffled_filenames)
            return shuffled_filenames

        self._process_filenames.append(_shuffle_filenames)
        return self

    def batch(self, batch_size: int, drop_remainder: bool = False):
        if batch_size is None:
            return self

        def _batch_filenames(filenames):
            return list(batch_sequence(filenames, batch_size, drop_remainder))

        def _include_batch_size(example: Dict):
            actual_batch_size = len(example['filename'])
            example['batch_size'] = actual_batch_size
            return example

        self._process_filenames.append(_batch_filenames)
        self._post_process.append(_include_batch_size)
        return self

    def take(self, take: int):
        def _take(filenames):
            return filenames[:take]

        self._process_filenames.append(_take)
        return self

    def get_example(self, idx: int):
        filename = self.filenames[idx]
        return load_single(filename)

    def __len__(self):
        return len(process_filenames(self.filenames, self._process_filenames))

    def pprint_example(self):
        pprint_example(self.get_example(0))


class CylindersDynamicsDatasetLoader:

    def __init__(self, dataset_dir):
        self.dataset_dirs = dataset_dir
        self.hparams = load_params(dataset_dir)
        self.scenario = None
        self.batch_metadata = {}

        self.pool = get_context("spawn").Pool()
        print(f"Created pool with {self.pool._processes} workers")

        self.data_collection_params = self.hparams['data_collection_params']
        self.scenario_params = self.data_collection_params.get('scenario_params', {})

        self.state_keys = self.data_collection_params['state_keys']
        self.state_keys.append('time_idx')
        self.state_metadata_keys = self.data_collection_params['state_metadata_keys']
        self.env_keys = self.data_collection_params['env_keys']
        self.action_keys = self.data_collection_params['action_keys']
        self.time_indexed_keys = self.state_keys + self.state_metadata_keys + self.action_keys

    def __del__(self):
        self.pool.terminate()

    def post_process(self, e):
        return e

    def get_datasets(self, mode: str, shuffle: Optional[int] = 0, take: int = None):
        filenames = get_filenames(self.dataset_dirs, mode)

        if len(filenames) == 0:
            raise EmptyDatasetException()

        dataset = NewBaseDataset(loader=self, filenames=filenames, mode=mode)
        if shuffle:
            dataset = dataset.shuffle(seed=shuffle)
        return dataset

    def index_time_batched(self, example_batched, t: int):
        e_t = numpify(remove_batch(index_time_batched(example_batched, self.time_indexed_keys, t, False)))
        return e_t


class MyTorchDataset(Dataset):

    def __init__(self,
                 dataset_dir: pathlib.Path,
                 mode: str,
                 transform=None,
                 no_update_with_metadata: bool = False):
        self.mode = mode
        self.dataset_dir = dataset_dir

        self.metadata_filenames = get_filenames(self.dataset_dir, mode)

        self.params = load_params(dataset_dir)

        self.transform = transform
        self.scenario = None
        self.no_update_with_metadata = no_update_with_metadata

    def __len__(self):
        return len(self.metadata_filenames)

    def __getitem__(self, idx):
        metadata_filename = self.metadata_filenames[idx]

        example = load_single(metadata_filename, no_update_with_metadata=self.no_update_with_metadata)

        if self.transform:
            example = self.transform(example)

        return example

    def get_scenario(self):
        if self.scenario is None:
            self.scenario = CylindersScenario()

        return self.scenario

    def pprint_example(self):
        pprint_example(self[0])
