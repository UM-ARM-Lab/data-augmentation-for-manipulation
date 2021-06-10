import random
from typing import List

from link_bot_data.dataset_utils import batch_sequence, merge_hparams_dicts, pprint_example
from link_bot_data.new_dataset_utils import load, get_filenames
from link_bot_pycommon.get_scenario import get_scenario


class NewBaseDataset:

    def __init__(self, loader, filenames: List):
        self.loader = loader
        self.filenames = filenames

    def __iter__(self):
        for filenames in self.filenames:
            e = load(filenames)
            e = self.loader.post_process(e)
            yield e

    def __len__(self):
        return len(self.filenames)

    def batch(self, batch_size: int, drop_remainder: bool = False):
        filenames_batched = list(batch_sequence(self.filenames, batch_size, drop_remainder))
        return NewBaseDataset(self.loader, filenames_batched)

    def shuffle(self):
        shuffled_filenames = self.filenames.copy()
        random.shuffle(shuffled_filenames)
        return NewBaseDataset(self.loader, shuffled_filenames)

    def take(self, take):
        return NewBaseDataset(self.loader, self.filenames[:take])


class NewBaseDatasetLoader:

    def __init__(self, dataset_dirs):
        self.dataset_dirs = dataset_dirs
        self.hparams = merge_hparams_dicts(dataset_dirs)
        self.scenario = None  # loaded lazily

    def post_process(self, e):
        return e

    def get_scenario(self):
        if self.scenario is None:
            self.scenario = get_scenario(self.hparams['scenario'])

        return self.scenario

    def get_datasets(self, mode: str, shuffle: bool = False, take: int = None):
        filenames = get_filenames(self.dataset_dirs, mode)
        assert len(filenames) > 0
        dataset = NewBaseDataset(self, filenames)
        if shuffle:
            dataset = dataset.shuffle()
        if take:
            dataset = dataset.take(take)
        return dataset

    def pprint_example(self):
        dataset = self.get_datasets(mode='val', take=1)
        example = next(iter(dataset))
        pprint_example(example)
