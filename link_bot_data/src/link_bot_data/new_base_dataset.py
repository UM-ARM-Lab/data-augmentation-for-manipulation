import random
from typing import List

from link_bot_data.dataset_utils import batch_sequence
from link_bot_data.new_dataset_utils import load


class NewBaseDataset:

    def __init__(self, filenames: List):
        self.filenames = filenames

    def __iter__(self):
        for filenames in self.filenames:
            yield load(filenames)

    def __len__(self):
        return len(self.filenames)

    def batch(self, batch_size: int, drop_remainder: bool = False):
        filenames_batched = list(batch_sequence(self.filenames, batch_size, drop_remainder))
        return NewBaseDataset(filenames_batched)

    def shuffle(self):
        shuffled_filenames = self.filenames.copy()
        random.shuffle(shuffled_filenames)
        return NewBaseDataset(shuffled_filenames)

    def take(self, take):
        return NewBaseDataset(self.filenames[:take])
