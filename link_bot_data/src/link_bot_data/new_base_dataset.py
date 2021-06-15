import pathlib
from multiprocessing import Pool
from typing import List, Dict, Optional, Callable

import numpy as np

from link_bot_data.dataset_utils import batch_sequence, merge_hparams_dicts, pprint_example
from link_bot_data.new_dataset_utils import load_possibly_batched, get_filenames, UNUSED_COMPAT
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization


class NewBaseDataset:

    def __init__(self, loader, filenames: List, post_process: Optional[List[Callable]] = None):
        self.loader = loader
        self.filenames = filenames
        self._post_process = post_process

    def __iter__(self):
        for filenames in self.filenames:
            e = load_possibly_batched(filenames, self.loader.loading_threadpool)
            e = self.loader.post_process(e)
            for p in self._post_process:
                e = p(e)
            yield e

    def __len__(self):
        return len(self.filenames)

    def batch(self, batch_size: int, drop_remainder: bool = False):
        filenames_batched = list(batch_sequence(self.filenames, batch_size, drop_remainder))

        def _add_batch(example: Dict):
            actual_batch_size = len(list(example.values())[0])
            example['batch_size'] = actual_batch_size
            return example

        # use self.__class__ here so that derived dataset classes return instances of themselves not the base class
        return self.__class__(self.loader, filenames_batched, [_add_batch])

    def shuffle(self, buffer_size=UNUSED_COMPAT, reshuffle_each_iteration=UNUSED_COMPAT):
        # FIXME: actually implementing this would be nice
        shuffled_filenames = self.filenames.copy()
        rng = np.random.RandomState(0)
        rng.shuffle(shuffled_filenames)
        return self.__class__(self.loader, shuffled_filenames, self._post_process)

    def take(self, take):
        return self.__class__(self.loader, self.filenames[:take], self._post_process)

    def map(self, _post_process: Callable):
        return self.__class__(self.loader, self.filenames, self._post_process + [_post_process])

    def prefetch(self, *args, **kwargs):
        return self


class NewBaseDatasetLoader:

    def __init__(self, dataset_dirs: List[pathlib.Path],
                 n_parallel=None,
                 scenario: Optional[ScenarioWithVisualization] = None):
        assert len(dataset_dirs) == 1
        self.dataset_dirs = dataset_dirs
        self.hparams = merge_hparams_dicts(dataset_dirs)
        self.n_parallel = n_parallel
        self.scenario = scenario
        self.batch_metadata = {}

        if self.n_parallel == 0:
            self.loading_threadpool = None
        else:
            self.loading_threadpool = Pool(processes=self.n_parallel)
            print(f"created threadpool with {self.loading_threadpool._processes} processes")

    def __del__(self):
        self.loading_threadpool.terminate()
        self.loading_threadpool.join()

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
