#!/usr/bin/env python
import json
import pathlib
from typing import Dict

import tensorflow as tf
import tensorflow.keras.layers as layers
from colorama import Fore
from tensorflow import keras
from tensorflow_probability import distributions as tfd

from link_bot_classifiers.base_recovery_actions_model import BaseRecoveryActionsModels
from link_bot_data.dataset_utils import add_predicted
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from moonshine import classifier_losses_and_metrics
from moonshine.get_local_environment import get_local_env_and_origin_2d_tf as get_local_env
from moonshine.moonshine_utils import add_batch, remove_batch, numpify
from moonshine.raster_2d import raster_2d
from moonshine.my_keras_model import MyKerasModel


class RNNRecoveryModel(MyKerasModel):
    def __init__(self, hparams: Dict, scenario: ExperimentScenario):
        super().__init__(hparams, None)
        self.scenario = scenario

        self.recovery_dataset_hparams = self.hparams['recovery_dataset_hparams']
        self.dynamics_dataset_hparams = self.recovery_dataset_hparams['fwd_model_hparams']['dynamics_dataset_hparams']
        self.n_action = self.dynamics_dataset_hparams['n_action']
        self.local_env_h_rows = self.hparams['local_env_h_rows']
        self.local_env_w_cols = self.hparams['local_env_w_cols']
        self.rope_image_k = self.hparams['rope_image_k']
        self.n_mixture_components = self.hparams['n_mixture_components']

        self.states_keys = self.hparams['states_keys']

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

        if self.hparams['batch_norm']:
            self.batch_norm = layers.BatchNormalization()

        self.env_state_encoder_dense_layers = []
        fc_layer_sizes = self.hparams['fc_layer_sizes'] + [self.hparams['rnn_size']]
        for hidden_size in fc_layer_sizes:
            dense = layers.Dense(hidden_size,
                                 activation='relu',
                                 kernel_regularizer=keras.regularizers.l2(self.hparams['kernel_reg']),
                                 bias_regularizer=keras.regularizers.l2(self.hparams['bias_reg']),
                                 activity_regularizer=keras.regularizers.l1(self.hparams['activity_reg']))
            self.env_state_encoder_dense_layers.append(dense)

        self.rnn = layers.LSTM(self.hparams['rnn_size'], unroll=True, return_sequences=True, return_state=True)

        self.n_covariance_parameters = int((self.n_action ** 2 + self.n_action) / 2)
        self.n_mean_parameters = self.n_action
        self.mixture_components_layer = layers.Dense(self.n_mixture_components, activation='softmax')
        self.means_layer = layers.Dense(self.n_mixture_components * self.n_mean_parameters, activation=None)
        self.covariances_layer = layers.Dense(self.n_mixture_components * self.n_covariance_parameters, activation=None)

        loss_type = self.hparams['loss_type']
        if loss_type == 'sequence':
            self.loss_function = classifier_losses_and_metrics.mdn_sequence_likelihood
        else:
            raise NotImplementedError()

    def compute_loss(self, dataset_element, outputs):
        return {
            'loss': self.loss_function(dataset_element, outputs)
        }

    @tf.function
    def make_trajectory_images(self,
                               environment,
                               start_states,
                               local_env_center_point,
                               batch_size,
                               ):
        """
        :arg: environment [B, H, W]
        :return: [batch, time, h, w, 1 + n_points]
        """
        # this will produce images even for "null" data,
        # but are masked out in the RNN, and not actually used in the computation
        local_env, local_env_origin = get_local_env(center_point=local_env_center_point,
                                                    full_env=environment['env'],
                                                    full_env_origin=environment['origin'],
                                                    res=environment['res'],
                                                    local_h_rows=self.local_env_h_rows,
                                                    local_w_cols=self.local_env_w_cols)

        concat_args = []
        for planned_state in start_states.values():
            planned_rope_image = raster_2d(state=planned_state,
                                           res=environment['res'],
                                           origin=local_env_origin,
                                           h=self.local_env_h_rows,
                                           w=self.local_env_w_cols,
                                           k=self.rope_image_k,
                                           batch_size=batch_size)
            concat_args.append(planned_rope_image)

        concat_args.append(tf.expand_dims(local_env, axis=3))
        images = tf.concat(concat_args, axis=3)
        return images

    def _conv(self, images, batch_size):
        conv_z = images
        for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
            conv_h = conv_layer(conv_z)
            conv_z = pool_layer(conv_h)
        out_conv_z = conv_z
        return tf.reshape(out_conv_z, [batch_size, -1])

    # @tf.function
    def sample(self, input_dict: Dict):
        batch_size = 1
        time = 5  # TODO: how to pick this number? can I use "stop tokens" instead?
        images, initial_h = self.state_encoder(input_dict, batch_size=batch_size, training=False)
        initial_c = tf.zeros([batch_size, self.hparams['rnn_size']], dtype=tf.float32)

        sampled_actions = []
        h_t = initial_h
        c_t = initial_c
        for t in range(time):
            component_weights_t, covariances_t, means_t = self.h_to_parameters(h_t, batch_size, time=1)

            mixture_density = self.mixture_distribution(alpha=component_weights_t, mu=means_t, sigma=covariances_t)
            sampled_action = mixture_density.sample()
            sampled_actions.append(sampled_action)

            out_h, h_t, c_t = self.rnn(inputs=sampled_action, training=False, initial_state=[h_t, c_t])

        sampled_actions = tf.concat(sampled_actions, axis=1)
        return sampled_actions

    @tf.function
    def call(self, input_dict: Dict, training, **kwargs):
        batch_size, time, _ = input_dict['action'].shape
        images, initial_h = self.state_encoder(input_dict, batch_size, training)
        component_weights_0, covariances_0, means_0 = self.h_to_parameters(initial_h, batch_size, 1)

        actions = input_dict['action']
        initial_c = tf.zeros([batch_size, self.hparams['rnn_size']], dtype=tf.float32)
        # the first element in the mask doesn't mean we should ignore the first action
        out_h, _, _ = self.rnn(inputs=actions, mask=input_dict['mask'], training=training,
                               initial_state=[initial_h, initial_c])

        # for every time step's output, map down to several vectors representing the parameters defining the mixture of gaussians
        # ignore the last output because it's not used in evaluating the likelihood of the action sequence
        component_weights, covariances, means = self.h_to_parameters(out_h[:, :-1], batch_size, time - 1)

        component_weights = tf.concat([component_weights_0, component_weights], axis=1)
        covariances = tf.concat([covariances_0, covariances], axis=1)
        means = tf.concat([means_0, means], axis=1)

        # FIXME: component weights should still cause the dense layers to have some gradient?
        means = tf.ones([batch_size, time, self.n_mixture_components, 2]) * 0.1
        covariances = tf.ones([batch_size, time, self.n_mixture_components, 2]) * 0.01
        mixture_density = self.mixture_distribution(alpha=component_weights, mu=means, sigma=covariances)
        valid_log_likelihood = self.gaussian_negative_log_likelihood(y=input_dict['action'],
                                                                     gm=mixture_density,
                                                                     mask=input_dict['mask'])
        return {
            'means': means,
            'covs': covariances,
            'alphas': component_weights,
            'images': images,
            'valid_log_likelihood': valid_log_likelihood
        }

    def compute_metrics(self, dataset_element, outputs):
        return {
            'max_mean': tf.reduce_max(outputs['means']),
            'min_mean': tf.reduce_min(outputs['means']),
            'max_cov': tf.reduce_max(outputs['covs']),
            'min_cov': tf.reduce_min(outputs['covs']),
        }

    def h_to_parameters(self, out_h, batch_size, time):
        means = tf.reshape(self.means_layer(out_h), [batch_size, time, self.n_mixture_components, -1])
        covariances = tf.reshape(self.covariances_layer(out_h), [batch_size, time, self.n_mixture_components, -1])
        component_weights = tf.reshape(self.mixture_components_layer(out_h),
                                       [batch_size, time, self.n_mixture_components, -1])
        return component_weights, covariances, means

    def state_encoder(self, input_dict, batch_size, training):
        # get only the start states
        start_state = {k: input_dict[add_predicted(k)][:, 0] for k in self.states_keys}
        # tile to the number of actions
        local_env_center_point = self.scenario.local_environment_center_differentiable(start_state)
        images = self.make_trajectory_images(environment=self.scenario.get_environment_from_example(input_dict),
                                             start_states=start_state,
                                             local_env_center_point=local_env_center_point,
                                             batch_size=batch_size)
        # import matplotlib.pyplot as plt
        # from matplotlib import cm
        # cmap = cm.viridis
        # out_image = state_image_to_cmap(images[0], cmap=cmap)
        # plt.imshow(out_image)
        # plt.show()
        conv_output = self._conv(images, batch_size)
        concat_args = [conv_output]
        for k, v in start_state.items():
            # note this assumes all state vectors are[x1,y1,...,xn,yn]
            points = tf.reshape(v, [batch_size, -1, 2])
            points = points - points[:, :, tf.newaxis, 0]
            v = tf.reshape(points, [batch_size, -1])
            concat_args.append(v)
        conv_output = tf.concat(concat_args, axis=1)
        if self.hparams['batch_norm']:
            conv_output = self.batch_norm(conv_output, training=training)
        z = conv_output
        for dense_layer in self.env_state_encoder_dense_layers:
            z = dense_layer(z)
        return images, z

    @staticmethod
    def mixture_distribution(alpha, mu, sigma):
        # scale_tril = tfp.math.fill_triangular(sigma)
        gm = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=tf.squeeze(alpha, 3)),
                                   components_distribution=tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma))
        return gm

    @staticmethod
    def gaussian_negative_log_likelihood(y, gm, mask):
        log_likelihood = gm.log_prob(y)
        valid_indices = tf.where(mask)
        valid_log_likelihood = tf.gather_nd(log_likelihood, valid_indices)
        return valid_log_likelihood


class RNNRecoveryModelWrapper(BaseRecoveryActionsModels):

    def __init__(self, path: pathlib.Path, scenario: ExperimentScenario):
        super().__init__(scenario)
        model_hparams_file = path.parent / 'params.json'
        self.model_hparams = json.load(model_hparams_file.open('r'))
        self.net = RNNRecoveryModel(hparams=self.model_hparams, scenario=scenario)
        self.ckpt = tf.train.Checkpoint(model=self.net)
        self.manager = tf.train.CheckpointManager(self.ckpt, path, max_to_keep=1)
        if self.manager.latest_checkpoint:
            print(Fore.CYAN + "Restored from {}".format(self.manager.latest_checkpoint) + Fore.RESET)
        self.ckpt.restore(self.manager.latest_checkpoint)

    def sample(self, environment: Dict, state: Dict):
        input_dict = environment
        input_dict.update({add_predicted(k): tf.expand_dims(v, axis=0) for k, v in state.items()})
        input_dict = add_batch(input_dict)
        input_dict = {k: tf.cast(v, tf.float32) for k, v in input_dict.items()}
        output = self.net.sample(input_dict)
        output = remove_batch(output)
        output = numpify(output)
        return output
