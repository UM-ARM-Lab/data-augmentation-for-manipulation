import pathlib

from torch.utils.data import Dataset

from cylinders_simple_demo.utils.cylinders_scenario import CylindersScenario
from cylinders_simple_demo.utils.data_utils import load_single, get_filenames, pprint_example
from cylinders_simple_demo.utils.utils import load_params


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
