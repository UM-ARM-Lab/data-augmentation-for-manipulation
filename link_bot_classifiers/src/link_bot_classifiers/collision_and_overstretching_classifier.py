import pathlib
from typing import List, Dict

import tensorflow as tf

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_classifiers.collision_checker_classifier import CollisionCheckerClassifier
from link_bot_classifiers.gripper_distance_classifier import GripperDistanceClassifier
from link_bot_pycommon.experiment_scenario import ExperimentScenario


class CollisionCheckerAndOverstretchingClassifier(BaseConstraintChecker):

    def __init__(self,
                 paths: List[pathlib.Path],
                 scenario: ExperimentScenario,
                 ):
        super().__init__(paths, scenario)
        self.cc = CollisionCheckerClassifier(paths, scenario)
        self.gd = GripperDistanceClassifier(paths, scenario)

    def check_constraint_tf(self,
                            environment: Dict,
                            states_sequence: List[Dict],
                            actions: List[Dict]):
        cc, _ = self.cc.check_constraint_tf(environment, states_sequence, actions)
        gd, _ = self.gd.check_constraint_tf(environment, states_sequence, actions)
        # gd | cc | gd * cc
        #  0 | 0  |    0
        #  1 | 0  |    0
        #  0 | 1  |    0
        #  1 | 1  |    1
        return gd * cc, tf.constant(0)
