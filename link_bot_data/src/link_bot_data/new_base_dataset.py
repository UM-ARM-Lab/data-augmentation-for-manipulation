import pathlib
from multiprocessing import get_context
from typing import List, Dict, Optional, Callable

import numpy as np
from tqdm import tqdm

from link_bot_data.dataset_utils import batch_sequence, merge_hparams_dicts, pprint_example, add_predicted
from link_bot_data.new_dataset_utils import get_filenames, UNUSED_COMPAT, load_single, EmptyDatasetException
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from moonshine.tensorflow_utils import batch_examples_dicts


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
        if self.n_prefetch is None or self.n_prefetch == 0:
            generator = self.iter_serial()
        else:
            generator = self.iter_multiprocessing()

        for example in generator:
            # NOTE: This post_process with both batched/non-batched inputs which is annoying
            example = self.loader.post_process(example)
            for p in self._post_process:
                example = p(example)

            yield example

    def iter_serial(self):
        print("Using slow, serial iteration")

        for filenames in process_filenames(self.filenames, self._process_filenames):
            if isinstance(filenames, list):
                examples_i = [load_single(metadata_filename_i) for metadata_filename_i in filenames]
                example = batch_examples_dicts(examples_i)
            else:
                example = load_single(filenames)

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

    def get_example(self, idx: int):
        filename = self.filenames[idx]
        return load_single(filename)

    def __len__(self):
        return len(process_filenames(self.filenames, self._process_filenames))

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

    def shuffle(self, buffer_size=UNUSED_COMPAT, seed: Optional[int] = 0, reshuffle_each_iteration=True):
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

    def skip(self, skip: int):
        def _skip(filenames: List[pathlib.Path]):
            return filenames[skip:]

        self._process_filenames.append(_skip)
        return self

    def shard(self, shard: int):
        def _shard(filenames: List[pathlib.Path]):
            return filenames[::shard]

        self._process_filenames.append(_shard)
        return self

    def take(self, take: int):
        def _take(filenames):
            return filenames[:take]

        self._process_filenames.append(_take)
        return self

    def map(self, _post_process: Callable):
        self._post_process.append(_post_process)
        return self

    def serial(self):
        self.n_prefetch = 0
        return self

    def prefetch(self, n_prefetch: int):
        if n_prefetch == -1:
            n_prefetch = 2
        self.n_prefetch = n_prefetch
        return self

    def pprint_example(self):
        for k, v in self.get_example(0).items():
            try:
                print(k, v.shape)
            except AttributeError:
                print(k, type(v))

    def parallel_map(self, f: Callable, *args, **kwargs):
        args = [(filename, f, args, kwargs) for filename in self.filenames]
        for e in tqdm(self.loader.pool.imap_unordered(map_func, args), total=len(self.filenames)):
            if e is not None:
                yield e

    def parallel_filter(self, f: Callable, *args, **kwargs):
        args = [(filename, f, args, kwargs) for filename in self.filenames]
        for e in tqdm(self.loader.pool.imap_unordered(filter_func, args), total=len(self.filenames)):
            if e is not None:
                yield e


def map_func(map_args):
    filename, f, args, kwargs = map_args
    example = load_single(filename)
    return f(example, *args, **kwargs)


def filter_func(map_args):
    filename, f, args, kwargs = map_args
    example = load_single(filename)
    if f(example, *args, **kwargs):
        return example
    return None


class NewBaseDatasetLoader:

    def __init__(self, dataset_dirs: List[pathlib.Path],
                 scenario: Optional[ScenarioWithVisualization] = None):
        self.dataset_dirs = dataset_dirs
        self.hparams = merge_hparams_dicts(dataset_dirs)
        self.scenario = scenario
        self.batch_metadata = {}

        self.pool = get_context("spawn").Pool()
        print(f"Created pool with {self.pool._processes} workers")

    def __del__(self):
        self.pool.terminate()

    def post_process(self, e):
        return e

    def get_scenario(self):
        if self.scenario is None:
            scenario_params = self.hparams['data_collection_params'].get('scenario_params', {})
            self.scenario = get_scenario(self.hparams['scenario'], params=scenario_params)

        return self.scenario

    def get_datasets(self, mode: str, shuffle: Optional[int] = 0, take: int = None):
        filenames = get_filenames(self.dataset_dirs, mode)

        if len(filenames) == 0:
            raise EmptyDatasetException()

        dataset = NewBaseDataset(loader=self, filenames=filenames, mode=mode)
        if shuffle:
            dataset = dataset.shuffle(seed=shuffle)
        if take:
            dataset = dataset.take(take)
        return dataset

    def pprint_example(self):
        dataset = self.get_datasets(mode='val', take=1)
        example = dataset.get_example(0)
        pprint_example(example)
