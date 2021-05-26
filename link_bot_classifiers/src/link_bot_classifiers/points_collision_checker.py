import pathlib
from typing import List, Dict, Optional

import tensorflow as tf

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_pycommon.collision_checking import batch_in_collision_tf_3d
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from moonshine.moonshine_utils import dict_of_sequences_to_sequence_of_dicts

DEFAULT_INFLATION_RADIUS = 0.00


def check_collision(scenario: ExperimentScenario,
                    environment: Dict,
                    state: Dict,
                    collision_check_object=True):
    if collision_check_object:
        points = scenario.state_to_points_for_cc(state)
    else:
        points = scenario.state_to_gripper_position(state)
    in_collision, _ = batch_in_collision_tf_3d(environment=environment,
                                               points=points,
                                               inflate_radius_m=DEFAULT_INFLATION_RADIUS)
    return in_collision


class PointsCollisionChecker(BaseConstraintChecker):

    def __init__(self,
                 path: pathlib.Path,
                 scenario: ExperimentScenario,
                 ):
        super().__init__(path, scenario)
        self.name = self.__class__.__name__
        self.local_h_rows = self.hparams['local_h_rows']
        self.local_w_cols = self.hparams['local_w_cols']
        self.local_c_channels = self.hparams['local_c_channels']
        self.horizon = 2
        self.data_collection_params = {
            'res': self.hparams['res']
        }

    def check_constraint_tf(self,
                            environment: Dict,
                            states_sequence: List[Dict],
                            actions):
        in_collision = False
        for t, state in enumerate(states_sequence):
            # DEBUGGING
            # self.scenario.plot_state_rviz(state, label='debugging', idx=t, color='yellow')
            in_collision = check_collision(self.scenario, environment, state)
            if in_collision:
                break
        constraint_satisfied = tf.cast(tf.logical_not(in_collision), tf.float32)[tf.newaxis]
        return constraint_satisfied

    def check_constraint_tf_batched(self,
                                    environment: Dict,
                                    states: Dict,
                                    actions: Dict,
                                    batch_size: int,
                                    state_sequence_length: int):
        # TODO: optimize this code
        environments_list = dict_of_sequences_to_sequence_of_dicts(environment)
        states_list = dict_of_sequences_to_sequence_of_dicts(states)
        c_s = []
        for b in range(batch_size):
            states_sequence = dict_of_sequences_to_sequence_of_dicts(states_list[b])
            c_b = self.check_constraint_tf(environments_list[b], states_sequence, actions)
            c_s.append(c_b)
        return tf.stack(c_s, axis=0)

    def check_constraint_from_example(self,
                                      example: Dict,
                                      training: Optional[bool] = False,
                                      batch_size: Optional[int] = 1,
                                      state_sequence_length: Optional[int] = 1,
                                      ):
        # NOTE: input will be batched
        # TODO: where should this come from?
        env_keys = ['env', 'res', 'origin', 'extent']
        state_keys = ['rope', 'left_gripper', 'right_gripper']
        action_keys = ['left_gripper_position', 'right_gripper_position']
        environment = {k: example[k] for k in env_keys}
        states = {k: example[k] for k in state_keys}
        actions = {k: example[k] for k in action_keys}

        return self.check_constraint_tf_batched(environment,
                                                states,
                                                actions,
                                                batch_size,
                                                state_sequence_length)

    def label_in_collision(self,
                           example: Dict,
                           batch_size: Optional[int] = 1,
                           state_sequence_length: Optional[int] = 1,
                           ):
        # NOTE: input will be batched and have time dimension
        # TODO: where should this come from?
        env_keys = ['env', 'res', 'origin', 'extent']
        state_keys = ['rope', 'left_gripper', 'right_gripper']
        action_keys = ['left_gripper_position', 'right_gripper_position']
        environment = {k: example[k] for k in env_keys}
        states = {k: example[k] for k in state_keys}

        # TODO: optimize this code
        environments_list = dict_of_sequences_to_sequence_of_dicts(environment)
        states_list = dict_of_sequences_to_sequence_of_dicts(states)
        in_collision = []
        for b in range(batch_size):
            states_sequence = dict_of_sequences_to_sequence_of_dicts(states_list[b])
            in_collision_b = []
            for t, state in enumerate(states_sequence):
                in_collision_t = bool(check_collision(self.scenario, environments_list[b], state))
                in_collision_b.append(in_collision_t)
            in_collision.append(in_collision_b)
        return tf.constant(in_collision, dtype=tf.float32)
