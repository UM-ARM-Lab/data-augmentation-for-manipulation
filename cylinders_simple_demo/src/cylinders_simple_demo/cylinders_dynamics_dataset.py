import pathlib
import pickle
from functools import lru_cache
from typing import OrderedDict

from torch.utils.data import Dataset

from cylinders_simple_demo.cylinders_scenario import CylindersScenario
from cylinders_simple_demo.data_utils import load_gzipped_pickle
from cylinders_simple_demo.numpify import numpify
from cylinders_simple_demo.utils import load_params, load_hjson


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
