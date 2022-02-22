import pathlib
import pickle
from typing import List, Union, Dict

from link_bot_data.dataset_utils import merge_hparams_dicts
from link_bot_pycommon.serialization import load_gzipped_pickle
from moonshine.filepath_tools import load_hjson, load_params

UNUSED_COMPAT = None


class EmptyDatasetException(Exception):
    pass


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
    metadata = load_metadata(metadata_filename)

    data_filename = metadata.pop("data")
    full_data_filename = metadata_filename.parent / data_filename
    example = load_gzipped_pickle(full_data_filename)
    example.update(metadata)
    example['metadata'] = metadata
    return example


def load_metadata(metadata_filename):
    if 'hjson' in metadata_filename.name:
        metadata = load_hjson(metadata_filename)
    elif 'pkl' in metadata_filename.name:
        with metadata_filename.open("rb") as f:
            metadata = pickle.load(f)
    else:
        raise NotImplementedError()
    metadata['filename'] = metadata_filename.stem
    metadata['full_filename'] = metadata_filename.as_posix()
    return metadata


class DynamicsDatasetParams:

    def __init__(self, dataset_dirs: Union[pathlib.Path, List[pathlib.Path]]):
        self.params = merge_hparams_dicts(dataset_dirs)
        self.dataset_dirs = dataset_dirs
        self.data_collection_params = self.params['data_collection_params']
        self.scenario_params = self.data_collection_params['scenario_params']
        self.state_description = self.data_collection_params['state_description']
        self.state_metadata_description = self.data_collection_params['state_metadata_description']
        self.action_description = self.data_collection_params['action_description']
        self.env_description = self.data_collection_params['env_description']
        self.state_keys = list(self.state_description.keys())
        self.state_keys.append('time_idx')
        self.state_metadata_keys = list(self.state_metadata_description.keys())
        self.env_keys = list(self.env_description.keys())
        self.action_keys = list(self.action_description.keys())
        self.time_indexed_keys = self.state_keys + self.state_metadata_keys + self.action_keys
