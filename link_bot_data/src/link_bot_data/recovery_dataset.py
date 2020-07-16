import pathlib
from typing import List

import tensorflow as tf

from link_bot_data.base_dataset import BaseDataset


class RecoveryDataset(BaseDataset):

    def __init__(self, dataset_dirs: List[pathlib.Path]):
        super(RecoveryDataset, self).__init__(dataset_dirs)
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
            'accept_probabilities'
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
        features_description = {}
        for feature_name in self.feature_names:
            features_description[feature_name] = tf.io.FixedLenFeature([], tf.string)

        return features_description

    def post_process(self, dataset: tf.data.TFRecordDataset, n_parallel_calls: int):
        def _add_unstuck_probability(example):
            n_accepts = tf.math.count_nonzero(example['accept_probabilities'][1] > 0.5)
            example['unstuck_probability'] = tf.cast(n_accepts / self.n_action_samples, tf.float32)
            return example

        dataset = dataset.map(_add_unstuck_probability)
        return dataset
