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
        assert len(states_sequence) == 2
        d = tf.linalg.norm(states_sequence[1]['right_gripper'] - states_sequence[1]['left_gripper'])
        not_too_far = d < self.max_d
        return tf.expand_dims(tf.cast(not_too_far, tf.float32), axis=0), tf.constant(0)
