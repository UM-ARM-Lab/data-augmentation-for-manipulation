from copy import deepcopy
from typing import Dict

import numpy as np
import tensorflow as tf
from pyjacobian_follower import JacobianFollower

from link_bot_pycommon.base_dual_arm_rope_scenario import follow_jacobian_from_example
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

        # TODO: use the new jacobian stuff so avoid having to connect to ROS things in this constructor
        self.jacobian_follower_no_cc = JacobianFollower(robot_namespace=self.scenario.robot_namespace,
                                                        translation_step_size=0.005,
                                                        minimize_rotation=True,
                                                        collision_check=False,
                                                        visualize=False)
        # self.jacobian_follower_no_cc.connect()

    def __call__(self, example: Dict, training: bool, **kwargs):
        out = self.net(example, training, **kwargs)
        example_np = numpify(example)
        reached, predicted_joint_positions = self.follow_jacobian_from_example(example_np)
        out['joint_positions'] = tf.convert_to_tensor(predicted_joint_positions, dtype=tf.float32)
        sequence_length = example[self.action_keys[0]].shape[1] + 1
        out['joint_names'] = tf.tile(example['joint_names'], [1, sequence_length, 1])
        # TODO: return reached somehow
        # out['reached'] = tf.convert_to_tensor(reached)
        return out

    def follow_jacobian_from_example(self, example: Dict):
        batch_size = example.pop("batch_size")
        tool_names = [self.scenario.robot.left_tool_name, self.scenario.robot.right_tool_name]
        preferred_tool_orientations = self.scenario.get_preferred_tool_orientations(tool_names)
        target_reached_batched = []
        pred_joint_positions_batched = []
        for b in range(batch_size):
            input_sequence_length = example[self.action_keys[0]].shape[1]
            target_reached = [True]
            example_b_t = index_batch_time(example, self.state_keys + self.action_keys, b, 0)
            pred_joint_positions = [example_b_t['joint_positions']]
            example_b_t['joint_names'] = example['joint_names'][b, 0]
            for t in range(input_sequence_length):
                example_b_t['left_gripper_position'] = example['left_gripper_position'][b, t]
                example_b_t['right_gripper_position'] = example['right_gripper_position'][b, t]
                _, reached_t, joint_positions_t = follow_jacobian_from_example(example_b_t,
                                                                               self.jacobian_follower_no_cc,
                                                                               tool_names,
                                                                               preferred_tool_orientations)
                example_b_t['joint_positions'] = joint_positions_t
                target_reached.append(reached_t)
                pred_joint_positions.append(joint_positions_t)
            target_reached_batched.append(target_reached)
            pred_joint_positions_batched.append(pred_joint_positions)

        pred_joint_positions_batched = np.array(pred_joint_positions_batched)
        target_reached_batched = np.array(target_reached_batched)
        return target_reached_batched, pred_joint_positions_batched


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