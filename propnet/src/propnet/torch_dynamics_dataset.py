import logging
import pathlib
import pickle
from typing import Dict

import torch
from torch.utils.data import Dataset

from link_bot_data.new_dataset_utils import get_filenames, load_single
from link_bot_pycommon.get_scenario import get_scenario
from moonshine.filepath_tools import load_params


def remove_keys(*keys):
    def _remove_keys(example):
        for k in keys:
            if k in example:
                example.pop(k)
        return example

    return _remove_keys


def add_stats_to_example(example: Dict, stats: Dict):
    for k, stats_k in stats.items():
        example[f'{k}/mean'] = stats_k[0]
        example[f'{k}/std'] = stats_k[1]
        example[f'{k}/n'] = stats_k[2]
    return example


class TorchDynamicsDataset(Dataset):

    def __init__(self, dataset_dir: pathlib.Path, mode: str, transform=None, add_stats=False):
        self.dataset_dir = dataset_dir
        self.metadata_filenames = get_filenames([dataset_dir], mode)
        self.add_stats = add_stats

        self.params = load_params(dataset_dir)

        self.transform = transform
        self.scenario = None

        self.data_collection_params = self.params['data_collection_params']
        self.state_keys = self.data_collection_params['state_keys']
        self.state_keys.append('time_idx')
        self.env_keys = self.data_collection_params['env_keys']
        self.action_keys = self.data_collection_params['action_keys']

        self.stats = None
        self.stats_filename = dataset_dir / 'stats.pkl'
        if self.stats_filename.exists():
            with self.stats_filename.open("rb") as f:
                self.stats = pickle.load(f)

    def __len__(self):
        return len(self.metadata_filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        metadata_filename = self.metadata_filenames[idx]
        example = load_single(metadata_filename)

        if self.add_stats:
            if self.stats is None:
                logging.getLogger(__file__).warning('no stats in this dataset?!')
            else:
                example = add_stats_to_example(example, self.stats)

        if self.transform:
            example = self.transform(example)

        return example

    def get_scenario(self):
        if self.scenario is None:
            self.scenario = get_scenario(self.params['scenario'])

        return self.scenario
