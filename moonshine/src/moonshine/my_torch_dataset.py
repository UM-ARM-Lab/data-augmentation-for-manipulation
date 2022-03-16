import pathlib

import numpy as np
from torch.utils.data import Dataset

from link_bot_data.dataset_utils import pprint_example, merge_hparams_dicts
from link_bot_data.new_dataset_utils import get_filenames, load_single
from link_bot_pycommon.get_scenario import get_scenario


class MyTorchDataset(Dataset):

    def __init__(self, dataset_dir: pathlib.Path, mode: str, transform=None):
        self.dataset_dir = dataset_dir
        self.mode = mode
        self.metadata_filenames = get_filenames([dataset_dir], mode)

        self.params = merge_hparams_dicts(dataset_dir)

        self.transform = transform
        self.scenario = None

    def __len__(self):
        return len(self.metadata_filenames)

    def __getitem__(self, idx):
        metadata_filename = self.metadata_filenames[idx]
        example = load_single(metadata_filename)

        if self.transform:
            example = self.transform(example)

        return example

    def get_scenario(self):
        if self.scenario is None:
            self.scenario = get_scenario(self.params['scenario'], self.scenario_params)

        return self.scenario

    def pprint_example(self):
        pprint_example(self[0])
