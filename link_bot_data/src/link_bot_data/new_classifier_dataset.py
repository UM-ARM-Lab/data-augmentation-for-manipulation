from typing import Dict

from link_bot_data.dataset_utils import merge_hparams_dicts, pprint_example, add_predicted
from link_bot_data.new_base_dataset import NewBaseDataset
from link_bot_data.new_dataset_utils import UNUSED_COMPAT, get_filenames
from link_bot_data.visualization import init_viz_env, init_viz_action, classifier_transition_viz_t
from link_bot_pycommon.get_scenario import get_scenario
from merrrt_visualization.rviz_animation_controller import RvizAnimation
from moonshine.indexing import index_time


class NewClassifierDatasetLoader:

    def __init__(self, dataset_dirs):
        self.dataset_dirs = dataset_dirs
        self.hparams = merge_hparams_dicts(dataset_dirs)
        self.scenario = None  # loaded lazily

        self.labeling_params = self.hparams['labeling_params']
        self.horizon = self.hparams['labeling_params']['classifier_horizon']
        self.true_state_keys = self.hparams['true_state_keys']
        self.state_metadata_keys = self.hparams['state_metadata_keys']
        self.predicted_state_keys = [add_predicted(k) for k in self.hparams['predicted_state_keys']]
        self.predicted_state_keys.append(add_predicted('stdev'))
        self.env_keys = self.hparams['env_keys']
        self.action_keys = self.hparams['action_keys']

    def get_scenario(self):
        if self.scenario is None:
            self.scenario = get_scenario(self.hparams['scenario'])

        return self.scenario

    def get_datasets(self,
                     mode: str,
                     shuffle_files: bool = False,
                     take: int = None,
                     do_not_process: bool = UNUSED_COMPAT,
                     slow: bool = UNUSED_COMPAT):
        filenames = get_filenames(self.dataset_dirs, mode)
        assert len(filenames) > 0
        dataset = NewBaseDataset(filenames)
        if shuffle_files:
            dataset = dataset.shuffle()
        if take:
            dataset = dataset.take(take)
        return dataset

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

    def pprint_example(self):
        dataset = self.get_datasets(mode='val', take=1)
        example = next(iter(dataset))
        pprint_example(example)
