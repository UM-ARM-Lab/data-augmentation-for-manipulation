import pathlib
from copy import deepcopy
from typing import Dict, List

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from pyjacobian_follower import JacobianFollower

from link_bot_pycommon.base_dual_arm_rope_scenario import follow_jacobian_from_example
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from moonshine.indexing import index_batch_time
from moonshine.moonshine_utils import sequence_of_dicts_to_dict_of_tensors, vector_to_dict, numpify
from moonshine.my_keras_model import MyKerasModel
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


class UnconstrainedDynamicsNN(MyKerasModel):

    def __init__(self, hparams: Dict, batch_size: int, scenario: ExperimentScenario):
        super().__init__(hparams=hparams, batch_size=batch_size)
        self.scenario = scenario
        self.initial_epoch = 0

        self.concat = layers.Concatenate()
        self.dense_layers = []
        for fc_layer_size in self.hparams['fc_layer_sizes']:
            self.dense_layers.append(layers.Dense(fc_layer_size, activation='relu', use_bias=True))

        self.state_keys: List = self.hparams['state_keys']
        self.action_keys: List = self.hparams['action_keys']
        dataset_state_description: Dict = self.hparams['dynamics_dataset_hparams']['state_description']
        self.dataset_action_description: Dict = self.hparams['dynamics_dataset_hparams']['action_description']
        self.state_description = {k: dataset_state_description[k] for k in self.state_keys}
        self.total_state_dimensions = sum([dataset_state_description[k] for k in self.state_keys])

        self.dense_layers.append(layers.Dense(self.total_state_dimensions, activation=None))

    @tf.function
    def call(self, example, training, mask=None):
        actions = {k: example[k] for k in self.action_keys}
        input_sequence_length = actions[self.action_keys[0]].shape[1]
        s_0 = {k: example[k][:, 0] for k in self.state_keys}

        pred_states = [s_0]
        for t in range(input_sequence_length):
            s_t = pred_states[-1]
            action_t = {k: example[k][:, t] for k in self.action_keys}
            local_action_t = self.scenario.put_action_local_frame(s_t, action_t)

            s_t_local = self.scenario.put_state_local_frame(s_t)
            states_and_actions = list(s_t_local.values()) + list(local_action_t.values())

            # concat into one big state-action vector
            z_t = self.concat(states_and_actions)
            for dense_layer in self.dense_layers:
                z_t = dense_layer(z_t)

            delta_s_t = vector_to_dict(self.state_description, z_t)
            s_t_plus_1 = self.scenario.integrate_dynamics(s_t, delta_s_t)

            pred_states.append(s_t_plus_1)

        pred_states_dict = sequence_of_dicts_to_dict_of_tensors(pred_states, axis=1)
        return pred_states_dict

    def compute_loss(self, example, outputs):
        return {
            'loss': self.scenario.dynamics_loss_function(example, outputs)
        }

    def compute_metrics(self, example, outputs):
        metrics = self.scenario.dynamics_metrics_function(example, outputs)
        metrics['loss'] = self.scenario.dynamics_loss_function(example, outputs)
        return metrics


class UDNNWrapper(BaseDynamicsFunction):

    def make_net_and_checkpoint(self, batch_size, scenario):
        net = UnconstrainedDynamicsNN(hparams=self.hparams, batch_size=batch_size, scenario=scenario)
        ckpt = tf.train.Checkpoint(model=net)
        return net, ckpt


class WithRobotKinematics:

    def __init__(self, base_model):
        self.base_model = base_model
        for k, v in dir(base_model):
            self.__setattr__(k, v)

    def __call__(self, example, training: bool, **kwargs):
        output: Dict = self.base_model(example, training, **kwargs)
        output['joint_positions'] = example['joint_positions'] + example['joint_positions_action']
        return output


class UDNNWithRobotKinematics:  # FIXME: inherit from something?

    def __init__(self, net: UnconstrainedDynamicsNN):
        self.net = net
        # copy things we need, the reason I'm not just doing net.__call__ = my_overload is that for some reason that
        # doesn't change net() from calling the TF __call__
        self.preprocess_no_gradient = net.preprocess_no_gradient
        self.state_keys = net.state_keys + ['joint_positions']
        self.action_keys = net.action_keys
        self.scenario = net.scenario

        self.j = JacobianFollower(robot_namespace=self.scenario.robot_namespace,
                                  translation_step_size=0.005,
                                  minimize_rotation=True,
                                  collision_check=False)

    def __call__(self, example: Dict, training: bool, **kwargs):
        out = self.net(example, training, **kwargs)
        example_np = numpify(example)
        _, predicted_joint_positions = self.follow_jacobian_from_example(example_np)
        out['joint_positions'] = tf.convert_to_tensor(predicted_joint_positions, dtype=tf.float32)
        sequence_length = example[self.state_keys[0]].shape[1] + 1
        out['joint_names'] = tf.tile(example['joint_names'], [1, sequence_length, 1])
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
            pred_joint_positions = [index_batch_time(example, ['joint_positions'], b, 0)['joint_positions']]
            for t in range(input_sequence_length):
                example_b_t = index_batch_time(example, self.state_keys + self.action_keys, b, t)
                example_b_t['joint_names'] = example['joint_names'][b, t]
                _, reached_t, joint_positions_t = follow_jacobian_from_example(example_b_t,
                                                                               self.j,
                                                                               tool_names,
                                                                               preferred_tool_orientations)
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

