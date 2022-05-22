import pathlib
from typing import Optional, Dict

from torch.utils.data import Dataset

from link_bot_data.dataset_utils import pprint_example, merge_hparams_dicts
from link_bot_data.new_dataset_utils import get_filenames, load_single, load_metadata
from link_bot_pycommon.get_scenario import get_scenario


class MyTorchDataset(Dataset):

    def __init__(self, dataset_dir: pathlib.Path, mode: str, transform=None, only_metadata=False,
                 is_empty: bool = False, no_update_with_metadata: bool = False):
        self.mode = mode
        self.dataset_dir = dataset_dir
        self.only_metadata = only_metadata
        if not is_empty:
            if isinstance(dataset_dir, list):
                dataset_dirs = dataset_dir
            else:
                dataset_dirs = [dataset_dir]
            self.metadata_filenames = get_filenames(dataset_dirs, mode)

        self.params = merge_hparams_dicts(dataset_dir)

        self.transform = transform
        self.scenario = None
        self.no_update_with_metadata = no_update_with_metadata

    def __len__(self):
        return len(self.metadata_filenames)

    def __getitem__(self, idx):
        metadata_filename = self.metadata_filenames[idx]
        if self.only_metadata:
            example = load_metadata(metadata_filename)
            return example

        example = load_single(metadata_filename, no_update_with_metadata=self.no_update_with_metadata)

        if self.transform:
            example = self.transform(example)

        return example

    def get_scenario(self, scenario_params: Optional[Dict] = None):
        if scenario_params is not None:
            _scenario_params = scenario_params
        elif hasattr(self, 'scenario_params'):
            _scenario_params = self.scenario_params
        else:
            _scenario_params = {}
        if self.scenario is None:
            self.scenario = get_scenario(self.params['scenario'], _scenario_params)

        return self.scenario

    def pprint_example(self):
        pprint_example(self[0])
