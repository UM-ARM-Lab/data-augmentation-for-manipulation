import pathlib

import torch
from torch.utils.data import Dataset

from link_bot_data.new_dataset_utils import get_filenames, load_single
from link_bot_pycommon.get_scenario import get_scenario
from moonshine.filepath_tools import load_params
from moonshine.moonshine_utils import to_list_of_strings


class TorchDynamicsDataset(Dataset):

    def __init__(self, dataset_dir: pathlib.Path, mode: str, transform=None):
        self.dataset_dir = dataset_dir
        self.metadata_filenames = get_filenames([dataset_dir], mode)

        self.params = load_params(dataset_dir)

        self.transform = transform
        self.scenario = None

    def __len__(self):
        return len(self.metadata_filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        metadata_filename = self.metadata_filenames[idx]
        example = load_single(metadata_filename)

        if 'joint_names' in example:
            example.pop('joint_names')

        if self.transform:
            example = self.transform(example)

        return example

    def get_scenario(self):
        if self.scenario is None:
            self.scenario = get_scenario(self.params['scenario'])

        return self.scenario
