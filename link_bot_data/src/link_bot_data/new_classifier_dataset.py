from typing import Dict, Optional

from link_bot_data.dataset_utils import add_predicted, add_label
from link_bot_data.new_base_dataset import NewBaseDatasetLoader
from link_bot_data.new_dataset_utils import UNUSED_COMPAT
from link_bot_data.visualization import init_viz_env, init_viz_action, classifier_transition_viz_t
from merrrt_visualization.rviz_animation_controller import RvizAnimation
from moonshine.indexing import index_time


class NewClassifierDatasetLoader(NewBaseDatasetLoader):

    def __init__(self, dataset_dirs, threshold: Optional[float] = None, load_true_states: bool = UNUSED_COMPAT):
        super().__init__(dataset_dirs)

        self.labeling_params = self.hparams['labeling_params']
        self.horizon = self.hparams['labeling_params']['classifier_horizon']
        self.true_state_keys = self.hparams['true_state_keys']
        self.state_metadata_keys = self.hparams['state_metadata_keys']
        self.predicted_state_keys = [add_predicted(k) for k in self.hparams['predicted_state_keys']]
        self.threshold = threshold if threshold is not None else self.labeling_params['threshold']
        self.predicted_state_keys.append(add_predicted('stdev'))
        self.env_keys = self.hparams['env_keys']
        self.action_keys = self.hparams['action_keys']

    def post_process(self, e):
        add_label(e, self.threshold)
        return e

    def get_datasets(self,
                     mode: str,
                     shuffle: bool = False,
                     take: int = None,
                     do_not_process: bool = UNUSED_COMPAT,
                     slow: bool = UNUSED_COMPAT):
        return super().get_datasets(mode, shuffle, take)

    def anim_transition_rviz(self, example: Dict):
        anim = RvizAnimation(scenario=self.scenario,
                             n_time_steps=self.horizon,
                             init_funcs=[init_viz_env, self.init_viz_action()],
                             t_funcs=[init_viz_env, self.classifier_transition_viz_t()])
        anim.play(example)

    def index_true_state_time(self, example: Dict, t: int):
        return index_time(example, self.true_state_keys, t, False)

    def index_pred_state_time(self, example: Dict, t: int):
        return index_time(example, self.predicted_state_keys, t, False)

    def classifier_transition_viz_t(self):
        return classifier_transition_viz_t(metadata={},
                                           state_metadata_keys=self.state_metadata_keys,
                                           predicted_state_keys=self.predicted_state_keys,
                                           true_state_keys=self.true_state_keys)

    def init_viz_action(self):
        return init_viz_action({}, self.action_keys, self.predicted_state_keys)
