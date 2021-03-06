import pathlib
from typing import List, Dict

import tensorflow as tf

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_pycommon.experiment_scenario import ExperimentScenario


class GripperDistanceChecker(BaseConstraintChecker):

    def __init__(self,
                 path: pathlib.Path,
                 scenario: ExperimentScenario,
                 ):
        super().__init__(path, scenario)
        self.max_d = self.hparams['max_distance_between_grippers']
        self.name = self.__class__.__name__

    def check_constraint_tf(self,
                            environment: Dict,
                            states_sequence: List[Dict],
                            actions):
        del environment  # unused
        too_far = False
        for state in states_sequence:
            d = tf.linalg.norm(state['right_gripper'] - state['left_gripper'])
            too_far = d > self.max_d
            if too_far:
                break
        return tf.expand_dims(tf.cast(tf.logical_not(too_far), tf.float32), axis=0)

    def check_constraint_tf_batched(self,
                                    environment: Dict,
                                    states: Dict,
                                    actions: Dict,
                                    batch_size: int,
                                    state_sequence_length: int):
        del environment  # unused
        d = tf.linalg.norm(states['right_gripper'] - states['left_gripper'], axis=-1)[:, -1]
        not_too_far = d < self.max_d
        return tf.expand_dims(tf.cast(not_too_far, tf.float32), axis=0)
