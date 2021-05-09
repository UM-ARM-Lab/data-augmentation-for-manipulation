import pathlib
from typing import List

import tensorflow as tf

from link_bot_data.base_dataset import BaseDatasetLoader
from link_bot_data.visualization import init_viz_env, init_viz_action, recovery_transition_viz_t
from link_bot_pycommon.get_scenario import get_scenario
from merrrt_visualization.rviz_animation_controller import RvizAnimation


def is_stuck(example):
    stuck = example['recovery_probability'][0] <= 0
    return stuck


def compute_recovery_probabilities(accept_probabilities, n_action_samples: int):
    n_accepts = tf.math.count_nonzero(accept_probabilities > 0.5, axis=1)
    return tf.cast(n_accepts / n_action_samples, tf.float32)


class RecoveryDatasetLoader(BaseDatasetLoader):

    def __init__(self, dataset_dirs: List[pathlib.Path]):
        super(RecoveryDatasetLoader, self).__init__(dataset_dirs)
        self.sorted = sorted
        self.scenario = get_scenario(self.hparams['scenario'])

        self.state_keys = self.hparams['state_keys']
        self.action_keys = self.hparams['action_keys']

        self.feature_names = [
            'env',
            'origin',
            'extent',
            'res',
            'traj_idx',
            'start_t',
            'end_t',
            'accept_probabilities',
        ]

        for k in self.hparams['state_keys']:
            self.feature_names.append(k)

        self.horizon = self.hparams["labeling_params"]["action_sequence_horizon"]
        self.n_action_samples = self.hparams["labeling_params"]["n_action_samples"]

        for k in self.state_keys:
            self.feature_names.append(k)

        for k in self.action_keys:
            self.feature_names.append(k)

        self.batch_metadata = {
            'time': 2
        }

    def make_features_description(self):
        features_description = super().make_features_description()
        for feature_name in self.feature_names:
            features_description[feature_name] = tf.io.FixedLenFeature([], tf.string)

        return features_description

    def post_process(self, dataset: tf.data.TFRecordDataset, n_parallel_calls: int):
        dataset = super().post_process(dataset, n_parallel_calls)

        dataset = dataset.map(self.add_recovery_probabilities)
        dataset = dataset.filter(is_stuck)
        return dataset

    def add_recovery_probabilities(self, example):
        example['recovery_probability'] = compute_recovery_probabilities(example['accept_probabilities'],
                                                                         self.n_action_samples)
        return example

    def anim_rviz(self, example):
        anim = RvizAnimation(scenario=self.scenario,
                             n_time_steps=self.horizon,
                             init_funcs=[init_viz_env,
                                         self.init_viz_action(),
                                         ],
                             t_funcs=[init_viz_env,
                                      self.recovery_transition_viz_t(),
                                      lambda s, e, t: self.scenario.plot_recovery_probability_t(e, t),
                                      ])
        anim.play(example)

    def recovery_transition_viz_t(self):
        return recovery_transition_viz_t(metadata=self.scenario_metadata, state_keys=self.state_keys)

    def init_viz_action(self):
        return init_viz_action(self.scenario_metadata, self.action_keys, self.state_keys)
