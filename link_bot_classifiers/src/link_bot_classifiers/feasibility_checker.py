import pathlib
from typing import List, Dict

import tensorflow as tf

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_pycommon.experiment_scenario import ExperimentScenario


class RobotFeasibilityChecker(BaseConstraintChecker):

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
        return tf.expand_dims(tf.cast(feasible, tf.float32), axis=0)


class FastRobotFeasibilityChecker(BaseConstraintChecker):

    def __init__(self,
                 path: pathlib.Path,
                 scenario: ExperimentScenario):
        super().__init__(path, scenario)

    def check_constraint_tf(self,
                            environment: Dict,
                            states_sequence: List[Dict],
                            actions: List[Dict]):
        constraint_satisfied = True
        for state, action, next_state in zip(states_sequence, actions, states_sequence[1:]):
            collision = self.scenario.is_moveit_robot_in_collision(environment=environment,
                                                                   state=next_state,
                                                                   action=action)
            feasible = not collision
            reached = self.scenario.moveit_robot_reached(state, action, next_state)
            constraint_satisfied = feasible and reached
            if not constraint_satisfied:
                break
        return tf.expand_dims(tf.cast(constraint_satisfied, tf.float32), axis=0)

    def check_constraint_tf_batched(self,
                                    environment: Dict,
                                    states: Dict,
                                    actions: Dict,
                                    batch_size: int,
                                    state_sequence_length: int):
        raise NotImplementedError()
