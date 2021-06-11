import random
from multiprocessing import Pool
from typing import List, Dict, Optional, Callable

from link_bot_data.dataset_utils import batch_sequence, merge_hparams_dicts, pprint_example
from link_bot_data.new_dataset_utils import load_possibly_batched, get_filenames, UNUSED_COMPAT
from link_bot_pycommon.get_scenario import get_scenario


class NewBaseDataset:

    def __init__(self, loader, filenames: List, post_process: Optional[List[Callable]] = None):
        self.loader = loader
        self.filenames = filenames
        self._post_process = post_process
        if self.loader.n_parallel == 0:
            self.loading_threadpool = None
        else:
            self.loading_threadpool = Pool(processes=self.loader.n_parallel)

    def __iter__(self):
        for filenames in self.filenames:
            e = load_possibly_batched(filenames, self.loading_threadpool)
            e = self.loader.post_process(e)
            for p in self._post_process:
                e = p(e)
            yield e

    def __len__(self):
        return len(self.filenames)

    def batch(self, batch_size: int, drop_remainder: bool = False):
        filenames_batched = list(batch_sequence(self.filenames, batch_size, drop_remainder))

        def _add_batch(example: Dict):
            example['batch_size'] = batch_size
            return example

        # use self.__class__ here so that derived dataset classes return instances of themselves not the base class
        return self.__class__(self.loader, filenames_batched, [_add_batch])

    def shuffle(self, buffer_size=UNUSED_COMPAT, reshuffle_each_iteration=UNUSED_COMPAT):
        # FIXME: actually implementing this would be nice
        shuffled_filenames = self.filenames.copy()
        random.shuffle(shuffled_filenames)
        return self.__class__(self.loader, shuffled_filenames, self._post_process)

    def take(self, take):
        return self.__class__(self.loader, self.filenames[:take], self._post_process)

    def prefetch(self, *args, **kwargs):
        return self


class NewBaseDatasetLoader:

    def __init__(self, dataset_dirs, n_parallel=None):
        self.dataset_dirs = dataset_dirs
        self.hparams = merge_hparams_dicts(dataset_dirs)
        self.n_parallel = n_parallel
        self.scenario = None  # loaded lazily
        self.batch_metadata = {}

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
