#!/usr/bin/env python
import json
import pathlib
from typing import Dict, List

import numpy as np
from matplotlib import cm
import rospy
import tensorflow as tf
from colorama import Fore
from moonshine.moonshine_utils import numpify, index_dict_of_batched_vectors_tf
from link_bot_pycommon.pycommon import log_scale_0_to_1
from link_bot_classifiers.base_recovery_policy import BaseRecoveryPolicy
from link_bot_data.link_bot_dataset_utils import NULL_PAD_VALUE, add_predicted
from jsk_recognition_msgs.msg import BoundingBox
from link_bot_pycommon import link_bot_sdf_utils
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.link_bot_sdf_utils import environment_to_occupancy_msg, compute_extent_3d, extent_to_env_size, batch_idx_to_point_3d_in_env_tf, batch_point_to_idx_tf_3d_in_batched_envs
from link_bot_pycommon.pycommon import make_dict_tf_float32
from link_bot_pycommon.rviz_animation_controller import RvizAnimationController
from moonshine.get_local_environment import get_local_env_and_origin_3d_tf as get_local_env
from moonshine.moonshine_utils import add_batch, remove_batch, sequence_of_dicts_to_dict_of_tensors
from moonshine.raster_3d import raster_3d
from mps_shape_completion_msgs.msg import OccupancyStamped
from shape_completion_training.my_keras_model import MyKerasModel
from tensorflow import keras
from tensorflow_core.python.keras import layers
from visualization_msgs.msg import Marker


class NNRecoveryModel(MyKerasModel):
    def __init__(self, hparams: Dict, batch_size: int, scenario: ExperimentScenario):
        super().__init__(hparams, batch_size)
        self.scenario = scenario

        self.debug_pub = rospy.Publisher('classifier_debug', OccupancyStamped, queue_size=10, latch=True)
        self.raster_debug_pub = rospy.Publisher('classifier_raster_debug', OccupancyStamped, queue_size=10, latch=True)
        self.local_env_bbox_pub = rospy.Publisher('local_env_bbox', BoundingBox, queue_size=10, latch=True)

        self.classifier_dataset_hparams = self.hparams['recovery_dataset_hparams']
        self.dynamics_dataset_hparams = self.classifier_dataset_hparams['fwd_model_hparams']['dynamics_dataset_hparams']
        self.local_env_h_rows = self.hparams['local_env_h_rows']
        self.local_env_w_cols = self.hparams['local_env_w_cols']
        self.local_env_c_channels = self.hparams['local_env_c_channels']
        self.rope_image_k = self.hparams['rope_image_k']

        self.state_keys = self.hparams['state_keys']
        self.action_keys = self.hparams['action_keys']

        self.conv_layers = []
        self.pool_layers = []
        for n_filters, kernel_size in self.hparams['conv_filters']:
            conv = layers.Conv3D(n_filters,
                                 kernel_size,
                                 activation='relu',
                                 kernel_regularizer=keras.regularizers.l2(self.hparams['kernel_reg']),
                                 bias_regularizer=keras.regularizers.l2(self.hparams['bias_reg']),
                                 trainable=False)
            pool = layers.MaxPool3D(self.hparams['pooling'])
            self.conv_layers.append(conv)
            self.pool_layers.append(pool)

        if self.hparams['batch_norm']:
            self.batch_norm = layers.BatchNormalization(trainable=True)

        self.dense_layers = []
        for hidden_size in self.hparams['fc_layer_sizes']:
            dense = layers.Dense(hidden_size,
                                 activation='relu',
                                 kernel_regularizer=keras.regularizers.l2(self.hparams['kernel_reg']),
                                 bias_regularizer=keras.regularizers.l2(self.hparams['bias_reg']),
                                 trainable=True)
            self.dense_layers.append(dense)

        self.output_layer1 = layers.Dense(128, activation='relu', trainable=True)
        self.output_layer2 = layers.Dense(1, activation=None, trainable=True)
        self.sigmoid = layers.Activation("sigmoid")

    def make_traj_voxel_grids_from_input_dict(self, input_dict: Dict, batch_size, time: int = 1):
        # Construct a [b, h, w, c, 3] grid of the indices which make up the local environment
        pixel_row_indices = tf.range(0, self.local_env_h_rows, dtype=tf.float32)
        pixel_col_indices = tf.range(0, self.local_env_w_cols, dtype=tf.float32)
        pixel_channel_indices = tf.range(0, self.local_env_c_channels, dtype=tf.float32)
        x_indices, y_indices, z_indices = tf.meshgrid(pixel_col_indices, pixel_row_indices, pixel_channel_indices)

        # Make batched versions for creating the local environment
        batch_y_indices = tf.cast(tf.tile(tf.expand_dims(y_indices, axis=0), [batch_size, 1, 1, 1]), tf.int64)
        batch_x_indices = tf.cast(tf.tile(tf.expand_dims(x_indices, axis=0), [batch_size, 1, 1, 1]), tf.int64)
        batch_z_indices = tf.cast(tf.tile(tf.expand_dims(z_indices, axis=0), [batch_size, 1, 1, 1]), tf.int64)

        # Convert for rastering state
        pixel_indices = tf.stack([y_indices, x_indices, z_indices], axis=3)
        pixel_indices = tf.expand_dims(pixel_indices, axis=0)
        pixel_indices = tf.tile(pixel_indices, [batch_size, 1, 1, 1, 1])

        # # DEBUG
        # # plot the occupancy grid
        # time_steps = np.arange(time)
        # anim = RvizAnimationController(time_steps)
        # b = 0
        # full_env_dict = {
        #     'env': input_dict['env'][b],
        #     'origin': input_dict['origin'][b],
        #     'res': input_dict['res'][b],
        #     'extent': input_dict['extent'][b],
        # }
        # self.scenario.plot_environment_rviz(full_env_dict)
        # # END DEBUG

        state = {k: input_dict[k][:, 0] for k in self.state_keys}

        local_env_center = self.scenario.local_environment_center_differentiable(state)
        # by converting too and from the frame of the full environment, we ensure the grids are aligned
        indices = batch_point_to_idx_tf_3d_in_batched_envs(local_env_center, input_dict)
        local_env_center = batch_idx_to_point_3d_in_env_tf(*indices, input_dict)

        local_env_t, local_env_origin_t = get_local_env(center_point=local_env_center,
                                                        full_env=input_dict['env'],
                                                        full_env_origin=input_dict['origin'],
                                                        res=input_dict['res'],
                                                        local_h_rows=self.local_env_h_rows,
                                                        local_w_cols=self.local_env_w_cols,
                                                        local_c_channels=self.local_env_c_channels,
                                                        batch_x_indices=batch_x_indices,
                                                        batch_y_indices=batch_y_indices,
                                                        batch_z_indices=batch_z_indices,
                                                        batch_size=batch_size)

        local_voxel_grid_t_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        local_voxel_grid_t_array = local_voxel_grid_t_array.write(0, local_env_t)
        for i, state_component_t in enumerate(state.values()):
            state_component_voxel_grid = raster_3d(state=state_component_t,
                                                   pixel_indices=pixel_indices,
                                                   res=input_dict['res'],
                                                   origin=local_env_origin_t,
                                                   h=self.local_env_h_rows,
                                                   w=self.local_env_w_cols,
                                                   c=self.local_env_c_channels,
                                                   k=self.rope_image_k,
                                                   batch_size=batch_size)

            local_voxel_grid_t_array = local_voxel_grid_t_array.write(i + 1, state_component_voxel_grid)
        local_voxel_grid_t = tf.transpose(local_voxel_grid_t_array.stack(), [1, 2, 3, 4, 0])
        # add channel dimension information because tf.function erases it somehow...
        local_voxel_grid_t.set_shape([None, None, None, None, len(self.state_keys) + 1])

        # # DEBUG
        # raster_dict = {
        #     'env': tf.clip_by_value(tf.reduce_max(local_voxel_grid_t[b][:, :, :, 1:], axis=-1), 0, 1),
        #     'origin': local_env_origin_t[b].numpy(),
        #     'res': input_dict['res'][b].numpy(),
        # }
        # raster_msg = environment_to_occupancy_msg(raster_dict, frame='local_occupancy')
        # local_env_dict = {
        #     'env': local_env_t[b],
        #     'origin': local_env_origin_t[b].numpy(),
        #     'res': input_dict['res'][b].numpy(),
        # }
        # msg = environment_to_occupancy_msg(local_env_dict, frame='local_occupancy')
        # link_bot_sdf_utils.send_occupancy_tf(self.scenario.broadcaster, local_env_dict, frame='local_occupancy')
        # self.debug_pub.publish(msg)
        # self.raster_debug_pub.publish(raster_msg)
        # # pred state

        # debugging_s_t = {k: input_dict[k][b, t] for k in self.state_keys}
        # self.scenario.plot_state_rviz(debugging_s_t, label='predicted', color='b')
        # # true state (not known to classifier!)
        # debugging_true_state_t = numpify({k: input_dict[k][b, t] for k in self.state_keys})
        # self.scenario.plot_state_rviz(debugging_true_state_t, label='actual')
        # # action
        # if t < time - 1:
        #     debuggin_action_t = numpify({k: input_dict[k][b, t] for k in self.action_keys})
        #     self.scenario.plot_action_rviz(debugging_s_t, debuggin_action_t)
        # local_extent = compute_extent_3d(*local_voxel_grid_t[b].shape[:3], resolution=input_dict['res'][b].numpy())
        # depth, width, height = extent_to_env_size(local_extent)
        # bbox_msg = BoundingBox()
        # bbox_msg.header.frame_id = 'local_occupancy'
        # bbox_msg.pose.position.x = width / 2
        # bbox_msg.pose.position.y = depth / 2
        # bbox_msg.pose.position.z = height / 2
        # bbox_msg.dimensions.x = width
        # bbox_msg.dimensions.y = depth
        # bbox_msg.dimensions.z = height
        # self.local_env_bbox_pub.publish(bbox_msg)

        # anim.step()
        # # END DEBUG

        conv_z = local_voxel_grid_t
        for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
            conv_h = conv_layer(conv_z)
            conv_z = pool_layer(conv_h)
        out_conv_z = conv_z
        out_conv_z_dim = out_conv_z.shape[1] * out_conv_z.shape[2] * out_conv_z.shape[3] * out_conv_z.shape[4]
        out_conv_z = tf.reshape(out_conv_z, [batch_size, out_conv_z_dim])
        return out_conv_z

    def compute_loss(self, dataset_element, outputs):
        y_true = dataset_element['recovery_probability'][:, 1:2]  # 1:2 instead of just 1 to preserve the shape
        y_pred = outputs['logits']
        loss = tf.keras.losses.binary_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=True)
        return {
            'loss': tf.reduce_mean(loss)
        }

    def calculate_metrics(self, dataset_element, outputs):
        y_true = dataset_element['recovery_probability'][:, 1]
        y_pred = tf.squeeze(outputs['probabilities'], axis=1)
        error = tf.reduce_mean(tf.math.abs(y_true - y_pred))
        return {
            'error': error
        }

    # @tf.function
    def call(self, input_dict: Dict, training, **kwargs):
        batch_size = input_dict['batch_size']

        conv_output = self.make_traj_voxel_grids_from_input_dict(input_dict, batch_size)

        state = {k: input_dict[k][:, 0] for k in self.state_keys}
        state_local = self.scenario.put_state_local_frame(state)
        action = {k: input_dict[k][:, 0] for k in self.action_keys}
        action = self.scenario.put_action_local_frame(state, action)
        concat_args = [conv_output] + list(state_local.values()) + list(action.values())
        concat_output = tf.concat(concat_args, axis=1)

        if self.hparams['batch_norm']:
            concat_output = self.batch_norm(concat_output, training=training)

        z = concat_output
        for dense_layer in self.dense_layers:
            z = dense_layer(z)
        out_h = z

        # for every timestep's output, map down to a single scalar, the logit for recovery probability
        out_h = self.output_layer1(out_h)
        logits = self.output_layer2(out_h)
        probabilities = self.sigmoid(logits)

        return {
            'logits': logits,
            'probabilities': probabilities,
        }


class NNRecoveryPolicy(BaseRecoveryPolicy):

    def __init__(self, hparams: Dict, model_dir: pathlib.Path, scenario: ExperimentScenario, rng: np.random.RandomState):
        super().__init__(hparams, model_dir, scenario, rng)
        self.model_dir = model_dir
        self.scenario = scenario

        # load the model?
        model_hparams_file = model_dir / 'params.json'
        if not model_hparams_file.exists():
            raise FileNotFoundError("no hparams file found!")
        self.model_hparams = json.load(model_hparams_file.open('r'))
        self.model = NNRecoveryModel(hparams=self.model_hparams, batch_size=1, scenario=self.scenario)
        self.ckpt = tf.train.Checkpoint(model=self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, model_dir / 'latest_checkpoint', max_to_keep=1)

        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print(Fore.CYAN + "Restored from {}".format(self.manager.latest_checkpoint) + Fore.RESET)
        else:
            raise RuntimeError("Failed to restore!!!")

        self.action_rng = np.random.RandomState(0)
        self.n_action_samples = self.hparams['n_action_samples']
        self.data_collection_params = self.hparams['recovery_dataset_hparams']['data_collection_params']

    def __call__(self, environment: Dict, state: Dict):
        # sample a bunch of actions (batched?) and pick the best one
        max_unstuck_probability = -1
        best_action = None
        # anim = RvizAnimationController(np.arange(self.n_action_samples))
        # while not anim.done:
        for _ in range(self.n_action_samples):
            self.scenario.last_action = None
            action = self.scenario.sample_action(environment=environment,
                                                 state=state,
                                                 data_collection_params=self.data_collection_params,
                                                 action_params=self.data_collection_params,
                                                 action_rng=self.action_rng)

            # TODO: use the unconstrained dynamics to predict the state resulting from (e, s, a)
            # then add that to the recovery_model_input

            recovery_model_input = environment
            recovery_model_input.update(add_batch(state))  # add time dimension to state and action
            recovery_model_input.update(add_batch(action))
            recovery_model_input = make_dict_tf_float32(add_batch(recovery_model_input))
            recovery_model_input.update({
                'batch_size': 1,
                'time': 2,
            })
            recovery_model_output = self.model(recovery_model_input, training=False)
            recovery_probability = recovery_model_output['probabilities']

            # self.scenario.plot_environment_rviz(environment)
            # self.scenario.plot_state_rviz(state, label='stuck state')
            self.scenario.plot_recovery_probability(recovery_probability)
            color_factor = log_scale_0_to_1(tf.squeeze(recovery_probability), k=100)
            self.scenario.plot_action_rviz(state, action, label='proposed', color=cm.Greens(color_factor), idx=1)

            if recovery_probability > max_unstuck_probability:
                max_unstuck_probability = recovery_probability
                print(max_unstuck_probability)
                best_action = action
                self.scenario.plot_action_rviz(state, action, label='best_proposed', color='g', idx=2)
            # anim.step()
        return best_action
