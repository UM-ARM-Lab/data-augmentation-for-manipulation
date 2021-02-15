import pathlib
from typing import List, Dict

import tensorflow as tf

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_pycommon.experiment_scenario import ExperimentScenario


class FeasibilityChecker(BaseConstraintChecker):

    def __init__(self,
                 path: pathlib.Path,
                 scenario: ExperimentScenario):
        super().__init__(path, scenario)

    def check_constraint_tf(self,
                            environment: Dict,
                            states_sequence: List[Dict],
                            actions: List[Dict]):
        assert len(states_sequence) == 2
        state = states_sequence[0]
        action = actions[0]
        feasible, predicted_joint_positions = self.scenario.is_motion_feasible(environment=environment,
                                                                               state=state,
                                                                               action=action)
        return tf.expand_dims(tf.cast(feasible, tf.float32), axis=0), tf.constant(0)
