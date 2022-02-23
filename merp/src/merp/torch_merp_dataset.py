import pathlib
from typing import Dict

import torch
from torch.utils.data import Dataset

from link_bot_data.dataset_utils import pprint_example, merge_hparams_dicts
from link_bot_data.new_dataset_utils import get_filenames, load_single
from link_bot_data.visualization import dynamics_viz_t, init_viz_env
from link_bot_pycommon.get_scenario import get_scenario
from merrrt_visualization.rviz_animation_controller import RvizAnimation
from moonshine.indexing import index_time_batched, index_time
from moonshine.numpify import numpify
from moonshine.torch_and_tf_utils import remove_batch


class TorchMERPDataset(Dataset):

    def __init__(self, dataset_dir: pathlib.Path, mode: str, transform=None):
        self.dataset_dir = dataset_dir
        self.mode = mode
        self.metadata_filenames = get_filenames([dataset_dir], mode)

        self.params = merge_hparams_dicts(dataset_dir)
        self.data_collection_params = self.params['data_collection_params']
        self.scenario_params = self.data_collection_params['scenario_params']
        self.state_description = self.data_collection_params['state_description']
        self.predicted_state_keys = self.data_collection_params['predicted_state_keys']
        self.state_metadata_description = self.data_collection_params['state_metadata_description']
        self.action_description = self.data_collection_params['action_description']
        self.env_description = self.data_collection_params['env_description']
        self.state_keys = list(self.state_description.keys())
        self.state_keys.append('time_idx')
        self.state_metadata_keys = list(self.state_metadata_description.keys())
        self.env_keys = list(self.env_description.keys())
        self.action_keys = list(self.action_description.keys())
        self.time_indexed_keys = self.state_keys + self.state_metadata_keys + self.action_keys
        self.time_indexed_keys_predicted = self.predicted_state_keys + self.state_metadata_keys + self.action_keys

        self.transform = transform
        self.scenario = None

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
            self.scenario = get_scenario(self.params['scenario'], self.scenario_params)

        return self.scenario

    def index_time_batched(self, example_batched, t: int):
        e_t = numpify(remove_batch(index_time_batched(example_batched, self.time_indexed_keys, t, False)))
        return e_t

    def index_time(self, example, t: int):
        e_t = numpify(index_time(example, self.time_indexed_keys, t, False))
        return e_t

    def pprint_example(self):
        pprint_example(self[0])

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
