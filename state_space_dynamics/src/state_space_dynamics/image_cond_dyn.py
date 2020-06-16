import pathlib
from typing import Dict, List

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from colorama import Fore
from tensorflow import keras

from link_bot_pycommon.experiment_scenario import ExperimentScenario
from moonshine.get_local_environment import get_local_env_and_origin_differentiable
from moonshine.image_functions import raster_differentiable
from moonshine.matrix_operations import batch_outer_product
from moonshine.moonshine_utils import add_batch, remove_batch, \
    dict_of_sequences_to_sequence_of_dicts_tf
from shape_completion_training.my_keras_model import MyKerasModel
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


class ImageCondDynamics(MyKerasModel):

    def __init__(self, hparams: Dict, batch_size: int, scenario: ExperimentScenario):
        super().__init__(hparams, batch_size)
        self.scenario = scenario
        self.initial_epoch = 0

        self.rope_image_k = self.hparams['rope_image_k']
        if not self.hparams['use_full_env']:
            self.local_env_h_rows = self.hparams['local_env_h_rows']
            self.local_env_w_cols = self.hparams['local_env_w_cols']
        self.batch_size = batch_size
        self.concat = layers.Concatenate()
        self.concat2 = layers.Concatenate()
        # State keys is all the things we want the model to take in/predict
        self.states_keys = self.hparams['states_keys']
        self.used_states_description = {}
        self.out_dim = 0
        # the states_description lists what's available in the dataset
        for available_state_name, n in self.hparams['dynamics_dataset_hparams']['states_description'].items():
            if available_state_name in self.states_keys:
                self.used_states_description[available_state_name] = n
                self.out_dim += n

        self.state_action_dense_layers = []
        if 'state_action_only_fc_layer_sizes' in self.hparams:
            for fc_layer_size in self.hparams['state_action_only_fc_layer_sizes']:
                self.state_action_dense_layers.append(layers.Dense(fc_layer_size, activation='relu'))

        self.conv_only_dense_layers = []
        if 'conv_only_fc_layer_sizes' in self.hparams:
            if len(self.hparams['conv_only_fc_layer_sizes']) > 0:
                for fc_layer_size in self.hparams['conv_only_fc_layer_sizes'][:-1]:
                    self.conv_only_dense_layers.append(layers.Dense(fc_layer_size, activation='relu'))
                self.conv_only_dense_layers.append(layers.Dense(self.hparams['conv_only_fc_layer_sizes'][-1], activation=None))

        self.final_dense_layers = []
        final_fc_layer_sizes = []
        if 'final_fc_layer_sizes' in self.hparams:
            final_fc_layer_sizes = self.hparams['final_fc_layer_sizes']
        elif 'fc_layer_sizes' in self.hparams:
            final_fc_layer_sizes = self.hparams['fc_layer_sizes']
        for fc_layer_size in final_fc_layer_sizes:
            self.final_dense_layers.append(layers.Dense(fc_layer_size, activation='relu'))
        self.final_dense_layers.append(layers.Dense(self.out_dim, activation=None))

        self.conv_layers = []
        self.pool_layers = []
        for n_filters, kernel_size in self.hparams['conv_filters']:
            conv = layers.Conv2D(n_filters,
                                 kernel_size,
                                 activation='relu',
                                 kernel_regularizer=keras.regularizers.l2(self.hparams['kernel_reg']),
                                 bias_regularizer=keras.regularizers.l2(self.hparams['bias_reg']),
                                 activity_regularizer=keras.regularizers.l1(self.hparams['activity_reg']))
            pool = layers.MaxPool2D(2)
            self.conv_layers.append(conv)
            self.pool_layers.append(pool)

        # How to combine the image hidden vector and state/action hidden vector
        combine_type = self.hparams.get('combine', 'concat')
        if combine_type == 'outer_product':
            def outer_product_then_flatten(args):
                state_action, conv_z_t = args
                outer_product = batch_outer_product(state_action, conv_z_t)
                return tf.reshape(outer_product, [self.batch_size, -1])

            self.combine_state_action_and_conv_output = layers.Lambda(outer_product_then_flatten)
        elif combine_type == 'concat':
            self.combine_state_action_and_conv_output = layers.Concatenate()

        self.flatten_conv_output = layers.Flatten()

    def compute_loss(self, dataset_element, outputs):
        return {
            'loss': self.scenario.dynamics_loss_function(dataset_element, outputs)
        }

    def calculate_metrics(self, dataset_element, outputs):
        return self.scenario.dynamics_metrics_function(dataset_element, outputs)

    @tf.function
    def get_local_env(self, center_point, full_env_origin, full_env, res):
        local_env, local_env_origin = get_local_env_and_origin_differentiable(center_point=center_point,
                                                                              full_env=full_env,
                                                                              full_env_origin=full_env_origin,
                                                                              res=res,
                                                                              local_h_rows=self.local_env_h_rows,
                                                                              local_w_cols=self.local_env_w_cols)
        return local_env, local_env_origin

    @tf.function
    def call(self, dataset_element, training, mask=None):
        input_dict, _ = dataset_element
        actions = input_dict['action']
        input_sequence_length = actions.shape[1]

        # Combine all the states into one big vector, based on which states were listed in the hparams file
        substates_0 = []
        for state_key, n in self.used_states_description.items():
            substate_0 = input_dict[state_key][:, 0]
            substates_0.append(substate_0)
        s_0 = tf.concat(substates_0, axis=1)

        # Remember everything this batched, but to keep things clear plural variable names will be reserved for sequences
        res = input_dict['full_env/res']
        full_env = input_dict['full_env/env']
        full_env_origin = input_dict['full_env/origin']

        pred_states = [s_0]
        for t in range(input_sequence_length):
            s_t = pred_states[-1]

            action_t = actions[:, t]

            if self.hparams['use_full_env']:
                env = full_env
                env_origin = full_env_origin
                env_h_rows = self.full_env_params.h_rows
                env_w_cols = self.full_env_params.w_cols
            else:
                state = self.state_vector_to_state_dict(s_t)
                local_env_center = self.scenario.local_environment_center_differentiable(state)
                # NOTE: we assume same resolution for local and full environment
                env, env_origin = self.get_local_env(local_env_center, full_env_origin, full_env, res)
                env_h_rows = self.local_env_h_rows
                env_w_cols = self.local_env_w_cols

            rope_image_t = raster_differentiable(s_t, res, env_origin, env_h_rows, env_w_cols, k=self.rope_image_k,
                                                 batch_size=self.batch_size)

            env = tf.expand_dims(env, axis=3)

            # CNN
            z_t = self.concat([rope_image_t, env])
            for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
                z_t = conv_layer(z_t)
                z_t = pool_layer(z_t)
            conv_z_t = self.flatten_conv_output(z_t)
            for dense_layer in self.conv_only_dense_layers:
                conv_z_t = dense_layer(conv_z_t)

            # state & action
            state_action_t = self.concat2([s_t, action_t])
            for dense_layer in self.state_action_dense_layers:
                state_action_t = dense_layer(state_action_t)

            # combine state/action vector with output of CNN
            full_z_t = self.combine_state_action_and_conv_output([state_action_t, conv_z_t])

            # dense layers for combined vector
            for dense_layer in self.final_dense_layers:
                full_z_t = dense_layer(full_z_t)

            if self.hparams['residual']:
                # residual prediction, otherwise just take the final hidden representation as the next state
                residual_t = full_z_t
                s_t_plus_1_flat = s_t + residual_t
            else:
                s_t_plus_1_flat = full_z_t

            pred_states.append(s_t_plus_1_flat)

        pred_states = tf.stack(pred_states, axis=1)

        # Split the stack of state vectors up by state name/dim
        output_states = self.state_vector_to_state_sequence_dict(pred_states)

        return output_states

    @tf.function
    def state_vector_to_state_dict(self, s_t):
        state_dict = {}
        start_idx = 0
        for state_key, n in self.used_states_description.items():
            end_idx = start_idx + n
            state_dict[state_key] = s_t[:, start_idx:end_idx]
            start_idx += n
        return state_dict

    @tf.function
    def state_vector_to_state_sequence_dict(self, pred_states):
        state_dict = {}
        start_idx = 0
        for state_key, n in self.used_states_description.items():
            end_idx = start_idx + n
            state_dict[state_key] = pred_states[:, :, start_idx:end_idx]
            start_idx += n
        return state_dict


class ImageCondDynamicsWrapper(BaseDynamicsFunction):

    def __init__(self, model_dir: pathlib.Path, batch_size: int, scenario: ExperimentScenario):
        super().__init__(model_dir, batch_size, scenario)
        self.net = ImageCondDynamics(hparams=self.hparams, batch_size=batch_size, scenario=scenario)
        self.states_keys = self.net.states_keys
        # find a way to convert the old checkpoints. This is necessary to restore old models.
        self.ckpt = tf.train.Checkpoint(model=self.net, net=self.net)
        # TODO: use shape_completion_training stuff here? ModelRunner?
        self.manager = tf.train.CheckpointManager(self.ckpt, model_dir, max_to_keep=1)
        status = self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print(Fore.CYAN + "Restored from {}".format(self.manager.latest_checkpoint) + Fore.RESET)
            if self.manager.latest_checkpoint:
                status.assert_existing_objects_matched()
        else:
            raise RuntimeError("Failed to restore!!!")

    def propagate_from_dataset_element(self, dataset_element, training=False):
        return self.net(dataset_element, training=training)

    def propagate_differentiable(self,
                                 environment: Dict,
                                 start_states: Dict[str, np.ndarray],
                                 actions: tf.Variable) -> List[Dict]:
        """
        :param full_env:        (H, W)
        :param full_env_origin: (2)
        :param res:             scalar
        :param start_states:          each value in the dictionary should be of shape (batch, n_state)
        :param actions:        (T, 2)
        :return: states:       each value in the dictionary should be a of shape [batch, T+1, n_state)
        """
        test_x = {
            # shape: T, 2
            'action': tf.convert_to_tensor(actions, dtype=tf.float32),
            # shape: H, W
            'full_env/env': tf.convert_to_tensor(environment['full_env/env'], dtype=tf.float32),
            # shape: 2
            'full_env/origin': tf.convert_to_tensor(environment['full_env/origin'], dtype=tf.float32),
            # scalar
            'full_env/res': tf.convert_to_tensor(environment['full_env/res'], dtype=tf.float32),
        }

        for state_key, v in start_states.items():
            # handles conversion from double -> float
            start_state = tf.convert_to_tensor(v, dtype=tf.float32)
            start_state_with_time_dim = tf.expand_dims(start_state, axis=0)
            test_x[state_key] = start_state_with_time_dim

        test_x = add_batch(test_x)
        predictions = self.net((test_x, False))
        predictions = remove_batch(predictions)
        predictions = dict_of_sequences_to_sequence_of_dicts_tf(predictions)

        return predictions


model = ImageCondDynamics
