import logging
import pathlib
from typing import Dict

import numpy as np
from torch.utils.data import DataLoader

from link_bot_data.new_dataset_utils import get_filenames, DynamicsDatasetParams, load_single
from link_bot_data.visualization import dynamics_viz_t, init_viz_env
from merrrt_visualization.rviz_animation_controller import RvizAnimation
from moonshine.indexing import index_time_batched, index_time
from moonshine.moonshine_utils import get_num_workers
from moonshine.my_torch_dataset import MyTorchDataset
from moonshine.numpify import numpify
from moonshine.torch_and_tf_utils import remove_batch
from moonshine.torch_datasets_utils import take_subset, my_collate

logger = logging.getLogger(__file__)


def remove_keys(*keys):
    def _remove_keys(example):
        for k in keys:
            example.pop(k, None)
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
                            collate_fn=my_collate,
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


class TorchDynamicsDataset(MyTorchDataset, DynamicsDatasetParams):

    def __init__(self, dataset_dir: pathlib.Path, mode: str, transform=None, add_stats=False):
        MyTorchDataset.__init__(self, dataset_dir, mode, transform)
        DynamicsDatasetParams.__init__(self, dataset_dir)
        self.dataset_dir = dataset_dir
        self.mode = mode
        self.metadata_filenames = get_filenames([dataset_dir], mode)
        self.add_stats = add_stats

    def get_datasets(self, mode=None):
        if mode != self.mode:
            raise RuntimeError("the mode must be set when constructing the Dataset, not when calling get_datasets")
        return TorchLoaderWrapped(dataset=self)

    def index_time_batched(self, example_batched, t: int):
        e_t = numpify(remove_batch(index_time_batched(example_batched, self.time_indexed_keys, t, False)))
        return e_t

    def index_time(self, example, t: int):
        e_t = numpify(index_time(example, self.time_indexed_keys, t, False))
        return e_t

    def dynamics_viz_t(self):
        return dynamics_viz_t(metadata={},
                              state_metadata_keys=self.state_metadata_keys,
                              state_keys=self.state_keys,
                              action_keys=self.action_keys)

    def anim_rviz(self, example: Dict):
        anim = RvizAnimation(self.get_scenario(),
                             n_time_steps=example['time_idx'].size,
                             init_funcs=[
                                 init_viz_env
                             ],
                             t_funcs=[
                                 init_viz_env,
                                 self.dynamics_viz_t()
                             ])
        anim.play(example)


def get_batch_size(batch):
    batch_size = len(batch['time_idx'])
    return batch_size


class TorchMetaDynamicsDataset(TorchDynamicsDataset):

    def __init__(self, dataset_dir: pathlib.Path, transform=None, eval_mode='val'):
        super().__init__(dataset_dir, mode='train', transform=transform)
        self.meta_metadata_filenames = get_filenames([dataset_dir], mode=eval_mode)

    def __getitem__(self, idx):
        train_metadata_filename = self.metadata_filenames[idx]
        train_example = load_single(train_metadata_filename)

        meta_example_idx = idx % len(self.meta_metadata_filenames)
        meta_train_metadata_filename = self.meta_metadata_filenames[meta_example_idx]
        meta_train_example = load_single(meta_train_metadata_filename)

        if self.transform:
            train_example = self.transform(train_example)
            meta_train_example = self.transform(meta_train_example)

        return {
            'train':      train_example,
            'meta_train': meta_train_example,
        }


def get_batch_size(batch):
    batch_size = len(batch['time_idx'])
    return batch_size
