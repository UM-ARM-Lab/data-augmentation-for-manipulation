from typing import Dict

from link_bot_data.new_base_dataset import NewBaseDatasetLoader
from link_bot_data.new_dataset_utils import UNUSED_COMPAT, DynamicsDatasetParams
from link_bot_data.visualization import init_viz_env, dynamics_viz_t
from merrrt_visualization.rviz_animation_controller import RvizAnimation
from moonshine.indexing import index_time_batched
from moonshine.torch_and_tf_utils import remove_batch
from moonshine.numpify import numpify


class NewDynamicsDatasetLoader(NewBaseDatasetLoader, DynamicsDatasetParams):

    def __init__(self, dataset_dirs):
        NewBaseDatasetLoader.__init__(self, dataset_dirs)
        DynamicsDatasetParams.__init__(self, dataset_dirs)

    def get_datasets(self,
                     mode: str,
                     shuffle: bool = False,
                     take: int = None,
                     do_not_process: bool = UNUSED_COMPAT,
                     slow: bool = UNUSED_COMPAT):
        return super().get_datasets(mode, shuffle, take)

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

    def index_time_batched(self, example_batched, t: int):
        e_t = numpify(remove_batch(index_time_batched(example_batched, self.time_indexed_keys, t, False)))
        return e_t
