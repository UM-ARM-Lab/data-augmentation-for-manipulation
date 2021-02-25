import pathlib
from typing import List, Dict

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
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    in_collision, _ = batch_in_collision_tf_3d(environment=environment,
                                               xs=xs,
                                               ys=ys,
                                               zs=zs,
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
        for state in states_sequence:
            in_collision = check_collision(self.scenario, environment, state)
            if in_collision:
                break
        constraint_satisfied = tf.cast(tf.logical_not(in_collision), tf.float32)[tf.newaxis]
        return constraint_satisfied, tf.ones([], dtype=tf.float32) * 1e-9

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
            state = dict_of_sequences_to_sequence_of_dicts(states_list[b])[1]
            c_b = check_collision(self.scenario, environments_list[b], state)
            c_s.append(c_b)
        return tf.stack(c_s, axis=0)[tf.newaxis], tf.ones([1, batch_size], dtype=tf.float32) * 1e-9
