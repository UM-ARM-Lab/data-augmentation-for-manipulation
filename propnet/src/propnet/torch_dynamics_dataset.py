import logging
import pathlib
from typing import Dict

import torch
from torch.utils.data import Dataset, DataLoader

from link_bot_data.new_dataset_utils import get_filenames, load_single
from link_bot_pycommon.get_scenario import get_scenario
from moonshine.filepath_tools import load_params
from moonshine.moonshine_utils import get_num_workers
from moonshine.torch_datasets_utils import take_subset

logger = logging.getLogger(__file__)


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


class TorchLoaderWrapped:
    """ this class is an attempt to make a pytorch dataset look like a NewBaseDataset objects """

    def __init__(self, dataset):
        self.dataset = dataset

    def take(self, take: int):
        dataset_subset = take_subset(self.dataset, take)
        return TorchLoaderWrapped(dataset=dataset_subset)

    def batch(self, batch_size: int):
        loader = DataLoader(dataset=self.dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=get_num_workers(batch_size=batch_size))
        for example in loader:
            actual_batch_size = list(example.values())[0].shape[0]
            example['batch_size'] = actual_batch_size
            yield example

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return iter(DataLoader(dataset=self.dataset,
                               batch_size=None,
                               shuffle=True,
                               num_workers=get_num_workers(batch_size=1)))


class TorchDynamicsDataset(Dataset):

    def __init__(self, dataset_dir: pathlib.Path, mode: str, transform=None, add_stats=False):
        self.dataset_dir = dataset_dir
        self.mode = mode
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

    def __len__(self):
        return len(self.metadata_filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        metadata_filename = self.metadata_filenames[idx]
        example = load_single(metadata_filename)

        if self.transform:
            example = self.transform(example)

        return example

    def get_scenario(self):
        if self.scenario is None:
            self.scenario = get_scenario(self.params['scenario'])

        return self.scenario

    def get_datasets(self, mode=None):
        if mode != self.mode:
            logger.warning("the mode must be set when constructing the Dataset, not when calling get_datasets")
        return TorchLoaderWrapped(dataset=self)


def get_batch_size(batch):
    batch_size = len(batch['time_idx'])
    return batch_size
