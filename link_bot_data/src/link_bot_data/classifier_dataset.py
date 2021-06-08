import pathlib
from typing import List, Dict, Optional

import tensorflow as tf
from colorama import Fore

import rospy
from link_bot_data.base_dataset import BaseDatasetLoader
from link_bot_data.dataset_utils import add_predicted, use_gt_rope, add_label, pprint_example
from link_bot_data.visualization import classifier_transition_viz_t, init_viz_action, init_viz_env
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from merrrt_visualization.rviz_animation_controller import RvizAnimation
from moonshine.indexing import index_time_with_metadata


class ClassifierDatasetLoader(BaseDatasetLoader):

    def __init__(self,
                 dataset_dirs: List[pathlib.Path],
                 load_true_states=False,
                 no_balance=True,
                 use_gt_rope: Optional[bool] = True,
                 threshold: Optional[float] = None,
                 old_compat: Optional[bool] = False,
                 verbose: int = 0,
                 scenario: Optional[ScenarioWithVisualization] = None,
                 ):
        super(ClassifierDatasetLoader, self).__init__(dataset_dirs, verbose)
        self.no_balance = no_balance
        self.load_true_states = load_true_states
        self.labeling_params = self.hparams['labeling_params']
        self.threshold = threshold if threshold is not None else self.labeling_params['threshold']
        self.use_gt_rope = use_gt_rope if use_gt_rope is not None else self.hparams['use_gt_rope']
        if not self.use_gt_rope:
            print(Fore.GREEN + f"NOT Using ground-truth rope" + Fore.RESET)
        print(f"classifier using threshold {self.threshold}")
        self.horizon = self.hparams['labeling_params']['classifier_horizon']
        if scenario is None:
            self.scenario = get_scenario(self.hparams['scenario'])
        else:
            self.scenario = scenario

        self.true_state_keys = self.hparams['true_state_keys']
        self.old_compat = old_compat
        if self.old_compat:
            self.true_state_keys.append('is_close')
        else:
            self.true_state_keys.append('error')

        self.state_metadata_keys = self.hparams['state_metadata_keys']
        self.predicted_state_keys = [add_predicted(k) for k in self.hparams['predicted_state_keys']]
        self.predicted_state_keys.append(add_predicted('stdev'))
        self.env_keys = self.hparams['env_keys']

        self.action_keys = self.hparams['action_keys']

        self.feature_names = [
            'classifier_start_t',
            'classifier_end_t',
            'traj_idx',
            'prediction_start_t',
        ]

        self.batch_metadata = {
            'time': self.horizon
        }

        if self.load_true_states:
            for k in self.true_state_keys:
                self.feature_names.append(k)

        for k in self.env_keys:
            self.feature_names.append(k)

        for k in self.state_metadata_keys:
            self.feature_names.append(k)

        for k in self.predicted_state_keys:
            self.feature_names.append(k)

        for k in self.action_keys:
            self.feature_names.append(k)

    def make_features_description(self):
        features_description = super().make_features_description()
        for feature_name in self.feature_names:
            features_description[feature_name] = tf.io.FixedLenFeature([], tf.string)

        return features_description

    def post_process(self, dataset: tf.data.TFRecordDataset, n_parallel_calls: int):
        dataset = super().post_process(dataset, n_parallel_calls)

        def _add_time(example: Dict):
            # this function is called before batching occurs, so the first dimension should be time
            example['time'] = tf.cast(self.horizon, tf.int64)
            return example

        # dataset = dataset.map(_add_time)
        # import numpy as np
        # def _debugging(example:Dict):
        #     example['origin_point'] = example['origin_point'] + np.array([0.01, 0.01, 0.01])
        #     return example
        #
        # dataset = dataset.map(_debugging)

        threshold = self.threshold

        def _label(example: Dict):
            add_label(example, threshold)
            return example

        if not self.old_compat:
            dataset = dataset.map(_label)

        if self.use_gt_rope:
            dataset = dataset.map(use_gt_rope)

        return dataset

    def anim_transition_rviz(self, example: Dict):
        anim = RvizAnimation(scenario=self.scenario,
                             n_time_steps=self.horizon,
                             init_funcs=[init_viz_env, self.init_viz_action()],
                             t_funcs=[init_viz_env, self.classifier_transition_viz_t()])
        anim.play(example)

    def index_true_state_time(self, example: Dict, t: int):
        return index_time_with_metadata(self.scenario_metadata, example, self.true_state_keys, t)

    def index_pred_state_time(self, example: Dict, t: int):
        return index_time_with_metadata(self.scenario_metadata, example, self.predicted_state_keys, t)

    def classifier_transition_viz_t(self):
        return classifier_transition_viz_t(metadata=self.scenario_metadata,
                                           state_metadata_keys=self.state_metadata_keys,
                                           predicted_state_keys=self.predicted_state_keys,
                                           true_state_keys=self.true_state_keys)

    def init_viz_action(self):
        return init_viz_action(self.scenario_metadata, self.action_keys, self.predicted_state_keys)

    def pprint_example(self):
        dataset = self.get_datasets(mode='val', take=1)
        example = next(iter(dataset))
        pprint_example(example)
