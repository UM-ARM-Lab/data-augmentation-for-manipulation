import pathlib
import random
from typing import List, Union

from link_bot_data.dataset_utils import merge_hparams_dicts, batch_sequence
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.serialization import load_gzipped_pickle
from moonshine.filepath_tools import load_hjson
from moonshine.moonshine_utils import batch_examples_dicts


def load_mode_filenames(d: pathlib.Path, filenames_filename: pathlib.Path):
    with filenames_filename.open("r") as filenames_file:
        filenames = [l.strip("\n") for l in filenames_file.readlines()]
    return [d / p for p in filenames]


def get_filenames(dataset_dirs, mode: str):
    all_filenames = []
    for d in dataset_dirs:
        if mode == 'all':
            all_filenames.extend(load_mode_filenames(d, d / f'train.txt'))
            all_filenames.extend(load_mode_filenames(d, d / f'test.txt'))
            all_filenames.extend(load_mode_filenames(d, d / f'val.txt'))
        else:
            filenames_filename = d / f'{mode}.txt'
            all_filenames.extend(load_mode_filenames(d, filenames_filename))
    all_filenames = sorted(all_filenames)
    return all_filenames


def load_single(metadata_filename: pathlib.Path):
    metadata = load_hjson(metadata_filename)
    data_filename = metadata.pop("data")
    full_data_filename = metadata_filename.parent / data_filename
    example = load_gzipped_pickle(full_data_filename)
    example.update(metadata)
    return example


def load(filenames: Union[pathlib.Path, List[pathlib.Path]]):
    if isinstance(filenames, list):
        examples_i = [load_single(metadata_filename_i) for metadata_filename_i in filenames]
        example = batch_examples_dicts(examples_i)
    else:
        metadata = load_hjson(filenames)
        data_filename = metadata.pop("data")
        full_data_filename = filenames.parent / data_filename
        example = load_gzipped_pickle(full_data_filename)
        example.update(metadata)
    return example


class NewDynamicsDataset:

    def __init__(self, filenames: List):
        self.filenames = filenames

    def __iter__(self):
        for filenames in self.filenames:
            yield load(filenames)

    def __len__(self):
        return len(self.filenames)

    def batch(self, batch_size: int):
        filenames_batched = list(batch_sequence(self.filenames, batch_size))
        return NewDynamicsDataset(filenames_batched)

    def shuffle(self):
        shuffled_filenames = self.filenames.copy()
        random.shuffle(shuffled_filenames)
        return NewDynamicsDataset(shuffled_filenames)

    def take(self, take):
        return NewDynamicsDataset(self.filenames[:take])


class NewDynamicsDatasetLoader:

    def __init__(self, dataset_dirs):
        self.dataset_dirs = dataset_dirs
        self.hparams = merge_hparams_dicts(dataset_dirs)
        self.scenario = None  # loaded lazily

    def get_scenario(self):
        if self.scenario is None:
            self.scenario = get_scenario(self.hparams['scenario'])

        return self.scenario

    def get_dataset(self, mode: str):
        filenames = get_filenames(self.dataset_dirs, mode)
        assert len(filenames) > 0
        return NewDynamicsDataset(filenames)
