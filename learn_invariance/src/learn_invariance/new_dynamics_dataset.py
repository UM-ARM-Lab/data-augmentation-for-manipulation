from typing import Dict

from link_bot_data.dataset_utils import merge_hparams_dicts
from link_bot_data.new_base_dataset import NewBaseDataset, NewBaseDatasetLoader
from link_bot_data.new_dataset_utils import get_filenames, UNUSED_COMPAT
from link_bot_data.visualization import dynamics_viz_t, init_viz_env
from link_bot_pycommon.get_scenario import get_scenario
from merrrt_visualization.rviz_animation_controller import RvizAnimation


class NewDynamicsDatasetLoader(NewBaseDatasetLoader):

    def __init__(self, dataset_dirs):
        super().__init__(dataset_dirs)

        self.data_collection_params = self.hparams['data_collection_params']
        self.steps_per_traj = self.data_collection_params['steps_per_traj']
        self.state_keys = self.data_collection_params['state_keys']
        self.state_metadata_keys = self.data_collection_params['state_metadata_keys']
        self.state_keys.append('time_idx')
        self.env_keys = self.data_collection_params['env_keys']
        self.action_keys = self.data_collection_params['action_keys']

    def get_datasets(self,
                     mode: str,
                     shuffle_files: bool = False,
                     take: int = None,
                     do_not_process: bool = UNUSED_COMPAT,
                     slow: bool = UNUSED_COMPAT):
        return super().get_datasets(mode, shuffle_files, take)

    def dynamics_viz_t(self):
        return dynamics_viz_t(metadata={},
                              state_metadata_keys=self.state_metadata_keys,
                              state_keys=self.state_keys,
                              action_keys=self.action_keys)

    def anim_rviz(self, example: Dict):
        anim = RvizAnimation(scenario=self.scenario,
                             n_time_steps=10,
                             init_funcs=[init_viz_env],
                             t_funcs=[init_viz_env, self.dynamics_viz_t()])
        anim.play(example)
