from copy import deepcopy
from typing import Dict

import numpy as np
import tensorflow as tf
from pyjacobian_follower import JacobianFollower

from moonshine.indexing import index_batch_time
from moonshine.moonshine_utils import numpify
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction
from state_space_dynamics.unconstrained_dynamics_nn import UnconstrainedDynamicsNN


# FIXME: inherit from BaseDynamicsFunction, but it's not that simple because the Ensemble/BaseDynamicsFunction
#  inheritance relationship is backwards?
class UDNNWithRobotKinematics:

    def __init__(self, net: UnconstrainedDynamicsNN):
        self.net = net
        # copy things we need, the reason I'm not just doing net.__call__ = my_overload is that for some reason that
        # doesn't change net() from calling the TF __call__
        self.preprocess_no_gradient = net.preprocess_no_gradient
        self.state_keys = net.state_keys + ['joint_positions']
        self.action_keys = net.action_keys
        self.scenario = net.scenario

        self.jacobian_follower_no_cc = JacobianFollower(robot_namespace=self.scenario.robot_namespace,
                                                        translation_step_size=0.005,
                                                        minimize_rotation=True,
                                                        collision_check=True,
                                                        visualize=False)

    def __call__(self, example: Dict, training: bool, **kwargs):
        scene_msg = example.pop("scene_msg")
        out = self.net(example, training, **kwargs)
        example_np = numpify(example)
        example_np['scene_msg'] = scene_msg
        reached, predicted_joint_positions = self.scenario.follow_jacobian_from_example(example_np)
        out['joint_positions'] = tf.convert_to_tensor(predicted_joint_positions, dtype=tf.float32)
        sequence_length = example[self.action_keys[0]].shape[1] + 1
        out['joint_names'] = tf.tile(example['joint_names'], [1, sequence_length, 1])
        # TODO: return reached somehow
        # out['reached'] = tf.convert_to_tensor(reached)
        return out


class UDNNWithRobotKinematicsWrapper(BaseDynamicsFunction):

    def make_net_and_checkpoint(self, batch_size, scenario):
        modified_hparams = deepcopy(self.hparams)
        modified_hparams['state_keys'].remove('joint_positions')
        net = UnconstrainedDynamicsNN(hparams=modified_hparams, batch_size=batch_size, scenario=scenario)
        ckpt = tf.train.Checkpoint(model=net)
        net = UDNNWithRobotKinematics(net)
        return net, ckpt

    def get_output_keys(self):
        return self.state_keys