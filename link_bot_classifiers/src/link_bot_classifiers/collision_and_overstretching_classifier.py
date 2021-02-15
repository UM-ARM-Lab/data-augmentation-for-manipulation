import pathlib
from typing import List, Dict

import tensorflow as tf

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_classifiers.points_collision_checker import PointsCollisionChecker
from link_bot_classifiers.gripper_distance_checker import GripperDistanceChecker
from link_bot_pycommon.experiment_scenario import ExperimentScenario


class CollisionCheckerAndOverstretchingClassifier(BaseConstraintChecker):

    def __init__(self,
                 paths: List[pathlib.Path],
                 scenario: ExperimentScenario,
                 ):
        super().__init__(paths, scenario)
        self.cc = PointsCollisionChecker(paths, scenario)
        self.gd = GripperDistanceChecker(paths, scenario)

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
