from copy import deepcopy
from typing import Dict

import tensorflow as tf
from colorama import Fore
from pyjacobian_follower import JacobianFollower

from moonshine.moonshine_utils import numpify
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction
from state_space_dynamics.unconstrained_dynamics_nn import UnconstrainedDynamicsNN


# FIXME: inherit from BaseDynamicsFunction, but it's not that simple because the Ensemble/BaseDynamicsFunction
#  inheritance relationship is backwards?
class UDNNWithRobotKinematics:

    def __init__(self, net: UnconstrainedDynamicsNN, collision_check: bool = True):
        self.net = net
        # copy things we need, the reason I'm not just doing net.__call__ = my_overload is that for some reason that
        # doesn't change net() from calling the TF __call__
        self.preprocess_no_gradient = net.preprocess_no_gradient
        self.state_keys = net.state_keys + ['joint_positions']
        self.action_keys = net.action_keys
        self.scenario = net.scenario
        self.collision_check = collision_check
        # NOTE: we need collision checking to be on here, because the FastRobotFeasibilityChecker relies on
        #  checking whether the tool positiosn match the actions.
        print(Fore.LIGHTBLUE_EX + f"{self.collision_check=}" + Fore.RESET)

        self.j_no_viz = JacobianFollower(robot_namespace=self.scenario.robot_namespace,
                                         translation_step_size=0.005,
                                         minimize_rotation=True,
                                         collision_check=self.collision_check,
                                         visualize=False)

    def __call__(self, inputs: Dict, training: bool, **kwargs):
        example_np, outputs = self.setup_inputs_for_jacobian_follower(inputs, kwargs, training)
        reached, joint_positions, joint_names = self.scenario.follow_jacobian_from_example(example_np, j=self.j_no_viz)
        self.add_robot_state_to_outputs(joint_names, joint_positions, outputs)
        return outputs

    def add_robot_state_to_outputs(self, joint_names, joint_positions, outputs):
        outputs['joint_positions'] = tf.convert_to_tensor(joint_positions, dtype=tf.float32)
        outputs['joint_names'] = joint_names

    def setup_inputs_for_jacobian_follower(self, example, kwargs, training):
        scene_msg = example["scene_msg"]
        net_example = {}
        for k, v in example.items():
            if k != 'scene_msg':
                net_example[k] = v
        out = self.net(net_example, training, **kwargs)
        example_np = numpify(net_example)  # this numpify is slow
        example_np['scene_msg'] = scene_msg
        return example_np, out


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
