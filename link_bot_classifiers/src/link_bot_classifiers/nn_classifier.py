import pathlib
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from colorama import Fore
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy, Metric

import rospy
from arc_utilities.ros_helpers import try_to_connect
from jsk_recognition_msgs.msg import BoundingBox
from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_data.dataset_utils import add_predicted, add_new
from link_bot_pycommon.bbox_visualization import grid_to_bbox
from link_bot_pycommon.debugging_utils import debug_viz_batch_indices
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.grid_utils import compute_extent_3d, batch_idx_to_point_3d_tf_res_origin_point
from link_bot_pycommon.grid_utils import send_voxelgrid_tf, batch_extent_to_origin_point_tf, environment_to_vg_msg, \
    vox_to_voxelgrid_stamped, send_voxelgrid_tf_origin_point_res_tf
from link_bot_pycommon.pycommon import make_dict_tf_float32
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from merrrt_visualization.rviz_animation_controller import RvizSimpleStepper, RvizAnimationController
from moonshine.classifier_losses_and_metrics import class_weighted_mean_loss
from moonshine.geometry import make_rotation_matrix_like, rotate_points_3d
from moonshine.get_local_environment import create_env_indices, get_local_env_and_origin_3d
from moonshine.metrics import BinaryAccuracyOnPositives, BinaryAccuracyOnNegatives, LossMetric, \
    FalsePositiveMistakeRate, FalseNegativeMistakeRate, FalsePositiveOverallRate, FalseNegativeOverallRate
from moonshine.moonshine_utils import add_batch, remove_batch, sequence_of_dicts_to_dict_of_tensors, numpify
from moonshine.my_keras_model import MyKerasModel
from moonshine.raster_3d import points_to_voxel_grid, points_to_voxel_grid_res_origin_point
from rviz_voxelgrid_visuals_msgs.msg import VoxelgridStamped
from std_msgs.msg import Float32
from visualization_msgs.msg import MarkerArray, Marker

DEBUG_INPUT = True
DEBUG_AUG = True
SHOW_ALL = False


class NNClassifier(MyKerasModel):
    def __init__(self, hparams: Dict, batch_size: int, scenario: ScenarioWithVisualization):
        super().__init__(hparams, batch_size)
        self.scenario = scenario
        self.broadcaster = self.scenario.tf.tf_broadcaster

        self.raster_debug_pubs = [
            rospy.Publisher(f'classifier_raster_debug_{i}', VoxelgridStamped, queue_size=10, latch=False) for i in
            range(5)]
        self.local_env_bbox_pub = rospy.Publisher('local_env_bbox', BoundingBox, queue_size=10, latch=True)
        self.env_aug_pub1 = rospy.Publisher("env_aug1", VoxelgridStamped, queue_size=10)
        self.env_aug_pub2 = rospy.Publisher("env_aug2", VoxelgridStamped, queue_size=10)
        self.env_aug_pub3 = rospy.Publisher("env_aug3", VoxelgridStamped, queue_size=10)
        self.env_aug_pub4 = rospy.Publisher("env_aug4", VoxelgridStamped, queue_size=10)
        self.env_aug_pub5 = rospy.Publisher("env_aug5", VoxelgridStamped, queue_size=10)
        self.object_state_pub = rospy.Publisher("object_state", VoxelgridStamped, queue_size=10)

        self.classifier_dataset_hparams = self.hparams['classifier_dataset_hparams']
        self.dynamics_dataset_hparams = self.classifier_dataset_hparams['fwd_model_hparams']['dynamics_dataset_hparams']
        self.true_state_keys = self.classifier_dataset_hparams['true_state_keys']
        self.pred_state_keys = [add_predicted(k) for k in self.classifier_dataset_hparams['predicted_state_keys']]
        self.pred_state_keys.append(add_predicted('stdev'))
        self.local_env_h_rows = self.hparams['local_env_h_rows']
        self.local_env_w_cols = self.hparams['local_env_w_cols']
        self.local_env_c_channels = self.hparams['local_env_c_channels']
        self.rope_image_k = self.hparams['rope_image_k']

        self.aug_hparams = self.hparams.get('augmentation', {})

        if self.aug_hparams.get('swept', True):
            self.is_env_augmentation_valid = NNClassifier.is_env_augmentation_valid_swept
        else:
            self.is_env_augmentation_valid = NNClassifier.is_env_augmentation_valid_discrete

        # TODO: add stdev to states keys?
        self.state_keys = self.hparams['state_keys']
        self.action_keys = self.hparams['action_keys']

        self.conv_layers = []
        self.pool_layers = []
        for n_filters, kernel_size in self.hparams['conv_filters']:
            conv = layers.Conv3D(n_filters,
                                 kernel_size,
                                 activation='relu',
                                 kernel_regularizer=keras.regularizers.l2(self.hparams['kernel_reg']),
                                 bias_regularizer=keras.regularizers.l2(self.hparams['bias_reg']))
            pool = layers.MaxPool3D(self.hparams['pooling'])
            self.conv_layers.append(conv)
            self.pool_layers.append(pool)

        if self.hparams['batch_norm']:
            self.batch_norm = layers.BatchNormalization()

        self.dense_layers = []
        for hidden_size in self.hparams['fc_layer_sizes']:
            dense = layers.Dense(hidden_size,
                                 activation='relu',
                                 kernel_regularizer=keras.regularizers.l2(self.hparams['kernel_reg']),
                                 bias_regularizer=keras.regularizers.l2(self.hparams['bias_reg']))
            self.dense_layers.append(dense)

        self.lstm = layers.LSTM(self.hparams['rnn_size'], unroll=True, return_sequences=True)
        self.output_layer = layers.Dense(1, activation=None)
        self.sigmoid = layers.Activation("sigmoid")

        self.certs_k = 100
        if self.hparams.get('uncertainty_head', False):
            self.uncertainty_head = keras.Sequential([layers.Dense(128, activation='relu'),
                                                      layers.Dense(128, activation='relu'),
                                                      layers.Dense(self.certs_k, activation=None),
                                                      ])

        self.aug_gen = tf.random.Generator.from_seed(0)
        self.aug_seed_stream = tfp.util.SeedStream(1, salt="nn_classifier_aug")
        self.aug_opt = tf.keras.optimizers.Adam(1e-4)

    def make_voxel_grid_inputs(self, input_dict: Dict, local_env, local_origin_point, batch_size, time):
        local_voxel_grids_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        for t in tf.range(time):
            state_t = {k: input_dict[add_predicted(k)][:, t] for k in self.state_keys}

            local_voxel_grid_t_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
            # Insert the environment as channel 0
            local_voxel_grid_t_array = local_voxel_grid_t_array.write(0, local_env)

            # insert the rastered states
            channel_idx = 0
            for channel_idx, (k, state_component_t) in enumerate(state_t.items()):
                points = tf.reshape(state_component_t, [batch_size, -1, 3])
                n_points_in_component = points.shape[1]
                flat_batch_indices = tf.repeat(tf.range(batch_size, dtype=tf.int64), n_points_in_component, axis=0)
                flat_points = tf.reshape(points, [-1, 3])
                flat_res = tf.repeat(input_dict['res'], n_points_in_component, axis=0)
                # flat_origin = tf.repeat(local_env_origin, n_points_in_component, axis=0)
                flat_origin_point = tf.repeat(local_origin_point, n_points_in_component, axis=0)
                state_component_voxel_grid = points_to_voxel_grid_res_origin_point(flat_batch_indices,
                                                                                   flat_points,
                                                                                   flat_res,
                                                                                   flat_origin_point,
                                                                                   self.local_env_h_rows,
                                                                                   self.local_env_w_cols,
                                                                                   self.local_env_c_channels,
                                                                                   batch_size)
                local_voxel_grid_t_array = local_voxel_grid_t_array.write(channel_idx + 1, state_component_voxel_grid)

            # insert the rastered robot state
            # could have the points saved to disc, load them up and transform them based on the current robot state?
            # (maybe by resolution? we could have multiple different resolutions)
            include_robot_voxels = self.hparams.get("include_robot_voxels", False)
            if include_robot_voxels:
                robot_voxel_grid = self.make_robot_voxel_grid(input_dict, t, batch_size)
                local_voxel_grid_t_array = local_voxel_grid_t_array.write(channel_idx + 1, robot_voxel_grid)
                n_channels = len(self.state_keys) + 2
            else:
                n_channels = len(self.state_keys) + 1

            local_voxel_grid_t = tf.transpose(local_voxel_grid_t_array.stack(), [1, 2, 3, 4, 0])
            # add channel dimension information because tf.function erases it somehow...
            local_voxel_grid_t.set_shape([None, None, None, None, n_channels])

            local_voxel_grids_array = local_voxel_grids_array.write(t, local_voxel_grid_t)

        return local_voxel_grids_array

    def conv_encoder(self, local_voxel_grids_aug_array: tf.TensorArray, batch_size, time):
        conv_outputs_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for t in tf.range(time):
            local_voxel_grid_t = local_voxel_grids_aug_array.read(t)

            out_conv_z = self.fwd_conv(batch_size, local_voxel_grid_t)

            conv_outputs_array = conv_outputs_array.write(t, out_conv_z)

        conv_outputs = conv_outputs_array.stack()
        conv_outputs = tf.transpose(conv_outputs, [1, 0, 2])
        return conv_outputs

    def get_local_env(self, batch_size, indices, input_dict):
        state_0 = {k: input_dict[add_predicted(k)][:, 0] for k in self.state_keys}

        # NOTE: to be more general, this should return a pose not just a point/position
        local_env_center = self.scenario.local_environment_center_differentiable(state_0)

        environment = {k: input_dict[k] for k in ['env', 'origin_point', 'res', 'extent']}
        local_env, local_origin_point = get_local_env_and_origin_3d(center_point=local_env_center,
                                                                    environment=environment,
                                                                    local_h_rows=self.local_env_h_rows,
                                                                    local_w_cols=self.local_env_w_cols,
                                                                    local_c_channels=self.local_env_c_channels,
                                                                    batch_x_indices=indices.batch_x,
                                                                    batch_y_indices=indices.batch_y,
                                                                    batch_z_indices=indices.batch_z,
                                                                    batch_size=batch_size)

        if DEBUG_AUG:
            stepper = RvizSimpleStepper()
            for b in debug_viz_batch_indices(self.batch_size):
                self.raster_debug_pubs[0].publish(vox_to_voxelgrid_stamped(local_env.numpy()[b],
                                                                           input_dict['res'][b].numpy(),
                                                                           frame='local_env_vg'))
                send_voxelgrid_tf_origin_point_res_tf(self.scenario.tf.tf_broadcaster,
                                                      local_origin_point[b],
                                                      input_dict['res'][b],
                                                      'local_env_vg')
                self.scenario.tf.send_transform(local_env_center[b], [0, 0, 0, 1], 'world', 'local_env_center',
                                                is_static=True)
                self.scenario.tf.send_transform(local_origin_point[b], [0, 0, 0, 1], 'world', 'local_origin_point',
                                                is_static=True)
                bbox_msg = grid_to_bbox(rows=self.local_env_h_rows,
                                        cols=self.local_env_w_cols,
                                        channels=self.local_env_c_channels,
                                        resolution=input_dict['res'][b].numpy())
                bbox_msg.header.frame_id = 'local_env_vg'
                self.local_env_bbox_pub.publish(bbox_msg)
                # stepper.step()

        return local_env, local_origin_point

    def sdf_opt_env_augmentation(self, example: Dict, indices, voxel_grids, add_aug, remove_aug):
        local_env_new, local_env_new_origin_point = self.get_new_local_env(indices, example)

        if DEBUG_AUG:
            stepper = RvizSimpleStepper()
            for b in debug_viz_batch_indices(self.batch_size):
                local_env_new_extent_b = compute_extent_3d(self.local_env_h_rows,
                                                           self.local_env_w_cols,
                                                           self.local_env_c_channels,
                                                           example['res'][b],
                                                           local_env_new_origin_point[b])
                local_env_new_dict = {
                    'env':          local_env_new[b].numpy(),
                    'origin_point': local_env_new_origin_point[b].numpy(),
                    'res':          example['res'][b].numpy(),
                    'extent':       local_env_new_extent_b,
                }
                msg = environment_to_vg_msg(local_env_new_dict, frame='local_env_new_vg', stamp=rospy.Time(0))
                # Show the new local environment we've sampled, in the place we sampled it
                self.env_aug_pub1.publish(msg)
                send_voxelgrid_tf(self.broadcaster, local_env_new_dict, frame='local_env_new_vg')

                # stepper.step()

                # Show the new local environment we've sampled, moved into the frame of the original local env,
                # the one we're augmenting
                msg2 = environment_to_vg_msg(local_env_new_dict, frame='local_env_vg', stamp=rospy.Time(0))
                self.env_aug_pub2.publish(msg2)

                # stepper.step()

        step_size = 1.0
        n_steps = 10
        p_e_nw = None
        for i in range(n_steps):
            add_sdf = compute_sdf(add_aug)
            remove_sdf = compute_sdf(remove_aug)
            add_gradient = compute_sdf_gradient(add_sdf)
            remove_gradient = compute_sdf_gradient(remove_sdf)
            sdf_gradient = add_gradient + remove_gradient  # [batch_size, num occupied voxels?]
            gradient = tf.reduce_mean(sdf_gradient, axis=-1)
            p_e_new = p_e_new + step_size * gradient
            e_new = apply_gradient_to_voxel_grid(e_new, p_e_new, gradient)

        return voxel_grids

    def merge_aug_and_local_voxel_grids(self,
                                        local_env_aug_masked,
                                        local_voxel_grids_array: tf.TensorArray,
                                        n_state_components,
                                        time):
        local_env_aug_expanded = tf.expand_dims(local_env_aug_masked, axis=-1)
        local_voxel_grids_aug_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        for t in tf.range(time):
            # the dims of local_voxel_grid are batch, rows, cols, channels, features
            # and features[0] is the local env, so pad the other feature dims with zero
            local_voxel_grid_t = local_voxel_grids_array.read(t)
            local_voxel_grid_states_t = local_voxel_grid_t[:, :, :, :, 1:]
            local_voxel_grid_aug_t = tf.concat([local_env_aug_expanded, local_voxel_grid_states_t], axis=-1)
            local_voxel_grid_aug_t = tf.clip_by_value(local_voxel_grid_aug_t, 0.0, 1.0)
            local_voxel_grids_aug_array = local_voxel_grids_aug_array.write(t, local_voxel_grid_aug_t)
        return local_voxel_grids_aug_array

    def get_new_local_env(self, indices, example):
        if add_new('env') not in example:
            print("new env not in example. did you forget the load from the pretransfer config?")
            example[add_new('env')] = example['env']
            example[add_new('extent')] = example['extent']
            example[add_new('origin_point')] = example['origin_point']
            example[add_new('res')] = example['res']

        new_env = {
            'env':          example[add_new('env')],
            'extent':       example[add_new('extent')],
            'origin_point': example[add_new('origin_point')],
            'res':          example[add_new('res')],
        }
        local_env_center = self.sample_local_env_position(new_env)
        local_env_new, local_origin_point_new = get_local_env_and_origin_3d(center_point=local_env_center,
                                                                            environment=new_env,
                                                                            local_h_rows=self.local_env_h_rows,
                                                                            local_w_cols=self.local_env_w_cols,
                                                                            local_c_channels=self.local_env_c_channels,
                                                                            batch_x_indices=indices.batch_x,
                                                                            batch_y_indices=indices.batch_y,
                                                                            batch_z_indices=indices.batch_z,
                                                                            batch_size=self.batch_size)
        return local_env_new, local_origin_point_new

    def is_env_augmentation_valid_swept(self,
                                        time,
                                        local_env_aug,
                                        local_voxel_grids_array: tf.TensorArray,
                                        n_state_components):
        raise NotImplementedError()

    def is_env_augmentation_valid_discrete(self, time, local_env_aug, local_voxel_grids_array: tf.TensorArray,
                                           n_state_components):
        aug_intersects_state_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for t in tf.range(time):
            local_voxel_grid_t = local_voxel_grids_array.read(t)
            local_voxel_grid_rastered_states_t = local_voxel_grid_t[:, :, :, :, 1:]
            states_flat_t = tf.reshape(local_voxel_grid_rastered_states_t, [self.batch_size, -1, n_state_components])
            aug_flat = tf.reshape(local_env_aug, [self.batch_size, -1])
            states_flat = tf.cast(tf.reduce_any(states_flat_t > 0.5, axis=-1), tf.float32)
            aug_intersects_state_t = tf.minimum(tf.reduce_sum(states_flat * aug_flat, axis=-1), 1.0)
            aug_intersects_state_array = aug_intersects_state_array.write(t, aug_intersects_state_t)

        # logical or of the validity at time t and t+1
        aug_intersects_state = aug_intersects_state_array.stack()
        aug_intersects_state = tf.transpose(aug_intersects_state, [1, 0])
        aug_intersects_state_any_t = tf.minimum(tf.reduce_sum(aug_intersects_state, axis=-1), 1.0)
        aug_is_valid = 1 - aug_intersects_state_any_t
        return aug_is_valid

    def sample_local_env_position(self, example):
        # NOTE: for my specific implementation of state_to_local_env_pose,
        #  sampling random states and calling state_to_local_env_pose is equivalent to sampling a point in the extent
        extent = tf.reshape(example['extent'], [self.batch_size, 3, 2])
        extent_lower = tf.gather(extent, 0, axis=-1)
        extent_upper = tf.gather(extent, 1, axis=-1)
        local_env_center = self.aug_gen.uniform([self.batch_size, 3], extent_lower, extent_upper)

        return local_env_center

    def create_env_indices(self, batch_size: int):
        return create_env_indices(self.local_env_h_rows, self.local_env_w_cols, self.local_env_c_channels, batch_size)

    def fwd_conv(self, batch_size, local_voxel_grid_t):
        conv_z = local_voxel_grid_t
        for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
            conv_h = conv_layer(conv_z)
            conv_z = pool_layer(conv_h)
        out_conv_z = conv_z
        out_conv_z_dim = out_conv_z.shape[1] * out_conv_z.shape[2] * out_conv_z.shape[3] * out_conv_z.shape[4]
        out_conv_z = tf.reshape(out_conv_z, [batch_size, out_conv_z_dim])
        return out_conv_z

    def compute_loss(self, dataset_element, outputs):
        # the labels are based on whether the predicted & observed rope states are close after the action, so t>=1
        is_close_after_start = dataset_element['is_close'][:, 1:]
        labels = tf.expand_dims(is_close_after_start, axis=2)
        logits = outputs['logits']
        bce = tf.keras.losses.binary_crossentropy(y_true=labels, y_pred=logits, from_logits=True)
        alpha = self.hparams.get('negative_label_weight', 0.5)
        if alpha != 0.5:
            print(Fore.YELLOW + f"Custom negative label weight = {alpha}")

        # alpha = 1 means ignore positive examples
        # alpha = 0 means ignore negative examples
        label_weight = tf.abs(is_close_after_start - alpha)
        label_weighted_bce = bce * label_weight

        # mask out / ignore examples where is_close [0] is 0
        if self.hparams.get('ignore_starts_far', False):
            print(Fore.YELLOW + "Ignoring starts-far")
            is_close_at_start = dataset_element['is_close'][:, 0]
            label_weighted_bce = label_weighted_bce * is_close_at_start

        # weight examples by the perception reliability, loss reliable means less important to predict, so lower loss
        if 'perception_reliability' in dataset_element and self.hparams.get('use_perception_reliability_loss', False):
            perception_reliability_weight = dataset_element['perception_reliability']
        else:
            perception_reliability_weight = tf.ones_like(label_weighted_bce)

        perception_reliability_weighted_bce = label_weighted_bce * perception_reliability_weight

        # mini-batches may not be balanced, weight the losses for positive and negative examples to balance
        total_bce = class_weighted_mean_loss(perception_reliability_weighted_bce, is_close_after_start)

        if self.hparams.get('uncertainty_head', False):
            # NOTE: loosely based on Tagasovska & Lopez-Paz, NeurIPS 2019
            uncertainty_loss = self.orthogonal_certificates_uncertainty_loss(outputs)
            total_loss = total_bce + uncertainty_loss
        else:
            total_loss = total_bce

        return {
            'loss': total_loss
        }

    def orthogonal_certificates_uncertainty_loss(self, outputs):
        w = self.uncertainty_head.weights[0]
        # loss = tf.reduce_max(tf.abs(outputs['uncertainty']))
        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true=tf.zeros_like(outputs['uncertainty']),
                                                                  y_pred=outputs['uncertainty'], from_logits=True))
        # diversity = tf.reduce_mean(tf.square(tf.matmul(tf.transpose(w), w) - tf.eye(self.certs_k)))
        return loss  # + diversity

    def create_metrics(self):
        super().create_metrics()
        return {
            'accuracy':              BinaryAccuracy(),
            'accuracy on negatives': BinaryAccuracyOnNegatives(),
            'accuracy on positives': BinaryAccuracyOnPositives(),
            'precision':             Precision(),
            'recall':                Recall(),
            'fp/mistakes':           FalsePositiveMistakeRate(),
            'fn/mistakes':           FalseNegativeMistakeRate(),
            'fp/total':              FalsePositiveOverallRate(),
            'fn/total':              FalseNegativeOverallRate(),
            # don't forget to include metrics for loss
            'loss':                  LossMetric(),
        }

    def compute_metrics(self, metrics: Dict[str, Metric], losses: Dict, dataset_element, outputs):
        labels = tf.expand_dims(dataset_element['is_close'][:, 1:], axis=2)
        probabilities = outputs['probabilities']
        metrics['accuracy'].update_state(y_true=labels, y_pred=probabilities)
        metrics['precision'].update_state(y_true=labels, y_pred=probabilities)
        metrics['recall'].update_state(y_true=labels, y_pred=probabilities)
        metrics['fp/mistakes'].update_state(y_true=labels, y_pred=probabilities)
        metrics['fn/mistakes'].update_state(y_true=labels, y_pred=probabilities)
        metrics['fp/total'].update_state(y_true=labels, y_pred=probabilities)
        metrics['fn/total'].update_state(y_true=labels, y_pred=probabilities)
        metrics['accuracy on negatives'].update_state(y_true=labels, y_pred=probabilities)
        metrics['accuracy on positives'].update_state(y_true=labels, y_pred=probabilities)

    def augmentation_optimization(self, input_dict, indices, local_origin_point, voxel_grids, batch_size, time):
        # # here, before augmenting anything, is where we define the constraints on the voxels of the environment
        # state_augmentation_type = self.hparams.get('state_type', None)
        # if state_augmentation_type != 'uniform':
        #     print("unsupported augmentation type")
        #     return voxel_grids

        # sample a translation and rotation for the object state
        translation, theta = self.scenario.sample_state_augmentation_variables(batch_size, self.aug_seed_stream)
        rotation = make_rotation_matrix_like(translation, theta)

        local_origin_point_aug = self.scenario.apply_state_augmentation(translation,
                                                                        rotation,
                                                                        input_dict,
                                                                        batch_size,
                                                                        time,
                                                                        self.local_env_h_rows,
                                                                        self.local_env_w_cols,
                                                                        self.local_env_c_channels)

        if DEBUG_AUG:
            self.debug_viz_local_env_pre_aug(input_dict, voxel_grids, time)

        # equivalent version of "logical or" over all state channels
        states_any_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for t in range(time):
            local_voxel_grid_t = voxel_grids.read(t)
            states_any_t = tf.minimum(tf.reduce_sum(local_voxel_grid_t[:, :, :, :, 1:], axis=-1), 1)
            states_any_array = states_any_array.write(t, states_any_t)
        states_any = tf.transpose(states_any_array.stack(), [1, 0, 2, 3, 4])
        states_any = tf.minimum(tf.reduce_sum(states_any, axis=1), 1)

        local_env = voxel_grids.read(0)[:, :, :, :, 0]  # just use 0 because it's the same at all time steps
        # the voxels where states_any == 1 and local_env == 1. In the future this will need to be swept volume,
        # and will also need to include the voxelized robot
        add = states_any * local_env

        # voxels where state_any == 1, and local_env == 0 must _not_ be occupied by the new env
        remove = states_any * (1 - local_env)

        if DEBUG_AUG:
            for b in debug_viz_batch_indices(batch_size):
                self.scenario.tf.send_transform(local_origin_point_aug[b], [0, 0, 0, 1], 'world',
                                                'local_origin_point_aug', is_static=True)

                add_dict = {
                    'env': add[b].numpy(),
                    'res': input_dict['res'][b].numpy(),
                }
                raster_msg = environment_to_vg_msg(add_dict, frame='local_env_vg', stamp=rospy.Time(0))
                self.env_aug_pub3.publish(raster_msg)

                remove_dict = {
                    'env': remove[b].numpy(),
                    'res': input_dict['res'][b].numpy(),
                }
                raster_msg = environment_to_vg_msg(remove_dict, frame='local_env_vg', stamp=rospy.Time(0))
                self.env_aug_pub4.publish(raster_msg)

        # TODO:
        # instead of figuring which voxels need to be on and converting those to points,
        # figure out which points in the object state are within res/2 of the points in the voxel grid
        # so that involves (1) converting the indices of the original local voxel grid into points in world frame
        # (2) selecting ones sufficiently close
        # (3) transforming those points, then translating to be in local_origin_point_aug frame

        # apply the transformation, giving two voxel grids that describe which voxels must be on/off for the new
        # augmented transition
        res = input_dict['res']
        remove_aug = self.transform_voxel_grid(remove, res, local_origin_point, local_origin_point_aug, translation,
                                               rotation, batch_size)
        add_aug = self.transform_voxel_grid(add, res, local_origin_point, local_origin_point_aug, translation, rotation,
                                            batch_size)

        if DEBUG_AUG:
            for b in debug_viz_batch_indices(batch_size):
                add_aug_dict = {
                    'env': add_aug[b].numpy(),
                    'res': input_dict['res'][b].numpy(),
                }
                raster_msg = environment_to_vg_msg(add_aug_dict, frame='local_env_aug_vg', stamp=rospy.Time(0))
                self.env_aug_pub3.publish(raster_msg)

                remove_aug_dict = {
                    'env': remove_aug[b].numpy(),
                    'res': input_dict['res'][b].numpy(),
                }
                raster_msg = environment_to_vg_msg(remove_aug_dict, frame='local_env_aug_vg', stamp=rospy.Time(0))
                self.env_aug_pub4.publish(raster_msg)

                send_voxelgrid_tf_origin_point_res_tf(self.scenario.tf.tf_broadcaster,
                                                      local_origin_point_aug[b],
                                                      input_dict['res'][b],
                                                      'local_env_aug_vg')

        e_aug = self.sdf_opt_env_augmentation(input_dict, indices, voxel_grids, add_aug, remove_aug)

        if DEBUG_AUG:
            stepper = RvizSimpleStepper()
            for b in debug_viz_batch_indices(batch_size):
                local_env_extent_b = compute_extent_3d(self.local_env_h_rows,
                                                       self.local_env_w_cols,
                                                       self.local_env_c_channels,
                                                       input_dict['res'][b],
                                                       local_env_origin[b])
                _aug_dict = {
                    'extent': local_env_extent_b,
                    'env':    voxel_grids.read(0)[b, :, :, :, 0].numpy(),
                    'origin': local_env_origin[b].numpy(),
                    'res':    input_dict['res'][b].numpy(),
                }
                msg = environment_to_vg_msg(_aug_dict, frame='local_env_vg', stamp=rospy.Time(0))
                self.env_aug_pub5.publish(msg)
                send_voxelgrid_tf(self.broadcaster, _aug_dict, frame='local_env_vg')

                # self.debug_viz_state_action(input_dict, b, 'input')

                # stepper.step()

        voxel_grids_aug = self.merge_aug_and_local_voxel_grids(e_aug,
                                                               voxel_grids,
                                                               n_state_components,
                                                               time)
        return voxel_grids_aug

    def transform_voxel_grid(self, voxel_grid, res, local_origin_point, local_origin_point_aug, translation, rotation,
                             batch_size):
        indices = np.where(voxel_grid > 0.5)
        indices = tf.stack(indices, axis=-1)
        batch_indices = indices[:, 0]
        indices_yxz = indices[:, 1:]
        points_res = tf.gather(res, batch_indices)
        points_origin_point = tf.gather(local_origin_point, batch_indices)
        points = batch_idx_to_point_3d_tf_res_origin_point(indices_yxz, points_res, points_origin_point)

        if DEBUG_AUG:
            try_to_connect(self.scenario.point_pub)
            for b in debug_viz_batch_indices(batch_size):
                points_b = tf.gather(points, tf.squeeze(tf.where(batch_indices == b)), axis=0)
                self.scenario.plot_points_rviz(points_b.numpy(), label='points_1', frame_id='world')

        # then apply translation/rotation to those points
        rotation_matrices = tf.gather(rotation, batch_indices)
        translation = tf.gather(translation, batch_indices)
        points_rotated = rotate_points_3d(rotation_matrices, points)
        points_aug_world_frame = points_rotated + translation

        if DEBUG_AUG:
            for b in debug_viz_batch_indices(batch_size):
                points_aug_b = tf.gather(points_aug_world_frame, tf.squeeze(tf.where(batch_indices == b)), axis=0)
                self.scenario.plot_points_rviz(points_aug_b.numpy(), label='points_aug_1', frame_id='world')

        # then shift so they're relative to local_origin_point_aug
        points_origin_points_aug = tf.gather(local_origin_point_aug, batch_indices)
        points_aug = points_aug_world_frame - points_origin_points_aug

        if DEBUG_AUG:
            for b in debug_viz_batch_indices(batch_size):
                points_aug_b = tf.gather(points_aug, tf.squeeze(tf.where(batch_indices == b)), axis=0)
                self.scenario.plot_points_rviz(points_aug_b.numpy(), label='points_aug_2',
                                               frame_id='local_origin_point_aug')

        # then convert back to a voxel grid
        voxel_grid = points_to_voxel_grid(batch_indices,
                                          points_aug,
                                          points_res,
                                          points_origin_points_aug,
                                          self.local_env_h_rows,
                                          self.local_env_w_cols,
                                          self.local_env_c_channels,
                                          batch_size)

        return voxel_grid

    def sample_p_e_new(self, input_dict):
        if add_new('env') not in input_dict:
            print("new env not in input_dict. did you forget the load from the pretransfer config?")
            input_dict[add_new('env')] = input_dict['env']
            input_dict[add_new('extent')] = input_dict['extent']
            input_dict[add_new('origin')] = input_dict['origin']
            input_dict[add_new('res')] = input_dict['res']

        new_env_dict = {
            'env':    input_dict[add_new('env')],
            'extent': input_dict[add_new('extent')],
            'origin': input_dict[add_new('origin')],
            'res':    input_dict[add_new('res')],
        }
        p_e_new = self.sample_local_env_position(new_env_dict)
        return p_e_new

    def call(self, input_dict: Dict, training, **kwargs):
        batch_size = input_dict['batch_size']
        time = tf.cast(input_dict['time'], tf.int32)

        input_dict['origin_point'] = batch_extent_to_origin_point_tf(input_dict['extent'], input_dict['res'])

        if DEBUG_INPUT:
            # clear the other voxel grids from previous calls
            vg_empty = np.zeros((64, 64, 64))
            empty_msg = vox_to_voxelgrid_stamped(vg_empty, scale=0.01, frame='world')
            for p in self.raster_debug_pubs:
                p.publish(empty_msg)
            self.env_aug_pub1.publish(empty_msg)
            self.env_aug_pub2.publish(empty_msg)
            self.env_aug_pub3.publish(empty_msg)
            self.env_aug_pub4.publish(empty_msg)
            self.env_aug_pub5.publish(empty_msg)

            stepper = RvizSimpleStepper()
            for b in debug_viz_batch_indices(batch_size):
                env_b = {
                    'env':          input_dict['env'][b],
                    'res':          input_dict['res'][b],
                    'origin_point': input_dict['origin_point'][b],
                    'extent':       input_dict['extent'][b],
                }
                self.scenario.plot_environment_rviz(env_b)
                self.delete_state_action_markers('aug')
                self.debug_viz_state_action(input_dict, b, 'input')
                origin_point_b = input_dict['origin_point'][b].numpy().tolist()
                self.scenario.tf.send_transform(origin_point_b, [0, 0, 0, 1], 'world', 'env_origin_point',
                                                is_static=True)
                # stepper.step()

        # Create voxel grids
        indices = self.create_env_indices(batch_size)
        local_env, local_origin_point = self.get_local_env(batch_size, indices, input_dict)

        voxel_grids = self.make_voxel_grid_inputs(input_dict, local_env, local_origin_point, batch_size, time)

        if training:
            # input_dict is also modified, but in place because it's a dict, where as voxel_grids is a tensor and
            # so modifying it internally won't change the value for the caller
            voxel_grids = self.augmentation_optimization(input_dict,
                                                         indices,
                                                         local_origin_point,
                                                         voxel_grids,
                                                         batch_size,
                                                         time)

        # encoder
        conv_output = self.conv_encoder(voxel_grids, batch_size=batch_size, time=time)
        out_h = self.fc(input_dict, conv_output, training)

        if self.hparams.get('uncertainty_head', False):
            out_uncertainty = self.uncertainty_head(out_h)

        # for every timestep's output, map down to a single scalar, the logit for accept probability
        all_accept_logits = self.output_layer(out_h)
        # ignore the first output, it is meaningless to predict the validity of a single state
        valid_accept_logits = all_accept_logits[:, 1:]
        valid_accept_probabilities = self.sigmoid(valid_accept_logits)

        outputs = {
            'logits':        valid_accept_logits,
            'probabilities': valid_accept_probabilities,
            'out_h':         out_h,
        }
        if self.hparams.get('uncertainty_head', False):
            outputs['uncertainty'] = out_uncertainty

        return outputs

    def fc(self, input_dict, conv_output, training):
        states = {k: input_dict[add_predicted(k)] for k in self.state_keys}
        states_in_local_frame = self.scenario.put_state_local_frame(states)
        actions = {k: input_dict[k] for k in self.action_keys}
        all_but_last_states = {k: v[:, :-1] for k, v in states.items()}
        actions = self.scenario.put_action_local_frame(all_but_last_states, actions)
        padded_actions = [tf.pad(v, [[0, 0], [0, 1], [0, 0]]) for v in actions.values()]
        if 'with_robot_frame' not in self.hparams:
            print("no hparam 'with_robot_frame'. This must be an old model!")
            concat_args = [conv_output] + list(states_in_local_frame.values()) + padded_actions
        elif self.hparams['with_robot_frame']:
            states_in_robot_frame = self.scenario.put_state_robot_frame(states)
            concat_args = ([conv_output] + list(states_in_robot_frame.values()) +
                           list(states_in_local_frame.values()) + padded_actions)
        else:
            concat_args = [conv_output] + list(states_in_local_frame.values()) + padded_actions
        if self.hparams['stdev']:
            stdevs = input_dict[add_predicted('stdev')]
            concat_args.append(stdevs)
        concat_output = tf.concat(concat_args, axis=2)
        if self.hparams['batch_norm']:
            concat_output = self.batch_norm(concat_output, training=training)
        z = concat_output
        for dense_layer in self.dense_layers:
            z = dense_layer(z)
        out_d = z
        out_h = self.lstm(out_d)
        return out_h

    def debug_viz_local_env_pre_aug(self, example: Dict, voxel_grids, time):
        for b in debug_viz_batch_indices(self.batch_size):
            bbox_msg = grid_to_bbox(rows=self.local_env_h_rows,
                                    cols=self.local_env_w_cols,
                                    channels=self.local_env_c_channels,
                                    resolution=example['res'][b].numpy())
            bbox_msg.header.frame_id = 'local_env'
            self.local_env_bbox_pub.publish(bbox_msg)

            send_voxelgrid_tf_origin_point_res_tf(self.broadcaster,
                                                  origin_point=example['origin_point'][b],
                                                  res=example['res'][b],
                                                  frame='local_env')

            self.animate_voxel_grid_states(b, example, voxel_grids, time)

    def animate_voxel_grid_states(self, b, example, voxel_grids, time):
        anim = RvizAnimationController(n_time_steps=time)
        while not anim.done:
            t = anim.t()

            local_voxel_grid_t = voxel_grids.read(t)

            for i, state_component_k_voxel_grid in enumerate(tf.transpose(local_voxel_grid_t, [4, 0, 1, 2, 3])):
                raster_dict = {
                    'env': tf.clip_by_value(state_component_k_voxel_grid[b], 0, 1),
                    'res': example['res'][b].numpy(),
                }
                raster_msg = environment_to_vg_msg(raster_dict, frame='local_env_vg', stamp=rospy.Time(0))
                self.raster_debug_pubs[i].publish(raster_msg)

            state_t = numpify({k: example[add_predicted(k)][b, t] for k in self.state_keys})
            error_msg = Float32()
            error_t = example['error'][b, 1]
            error_msg.data = error_t
            self.scenario.plot_state_rviz(state_t)
            self.scenario.plot_is_close(example['is_close'][b, 1])
            self.scenario.error_pub.publish(error_msg)

            anim.step()

    def delete_state_action_markers(self, label):
        def _make_delete_marker(ns):
            delete_marker = Marker()
            delete_marker.action = Marker.DELETEALL
            delete_marker.ns = ns
            return delete_marker

        state_delete_msg = MarkerArray(markers=[_make_delete_marker(label + '_l'),
                                                _make_delete_marker(label + 'aug_r'),
                                                _make_delete_marker(label + 'aug_rope')])
        self.scenario.state_viz_pub.publish(state_delete_msg)
        action_delete_msg = MarkerArray(markers=[_make_delete_marker(label)])
        self.scenario.action_viz_pub.publish(action_delete_msg)

    def debug_viz_state_action(self, input_dict, b, label: str, color='red'):
        state_0 = numpify({k: input_dict[add_predicted(k)][b, 0] for k in self.state_keys})
        action_0 = numpify({k: input_dict[k][b, 0] for k in self.action_keys})
        state_1 = numpify({k: input_dict[add_predicted(k)][b, 1] for k in self.state_keys})
        error_msg = Float32()
        error_t = input_dict['error'][b, 1]
        error_msg.data = error_t
        self.scenario.plot_state_rviz(state_0, idx=0, label=label, color=color)
        self.scenario.plot_state_rviz(state_1, idx=1, label=label, color=color)
        self.scenario.plot_action_rviz(state_0, action_0, idx=1, label=label, color=color)
        self.scenario.plot_is_close(input_dict['is_close'][b, 1])
        self.scenario.error_pub.publish(error_msg)


class NNClassifierWrapper(BaseConstraintChecker):
    def __init__(self, path: pathlib.Path, batch_size: int, scenario: ExperimentScenario):
        """
        Unlike the BaseConstraintChecker, this takes in list of paths, like cl_trials/dir1/dir2/best_checkpoint
        Args:
            path:
            batch_size:
            scenario:
        """
        super().__init__(path, scenario)
        self.name = self.__class__.__name__
        # FIXME: Bad API design
        assert isinstance(scenario, ScenarioWithVisualization)

        self.dataset_labeling_params = self.hparams['classifier_dataset_hparams']['labeling_params']
        self.data_collection_params = self.hparams['classifier_dataset_hparams']['data_collection_params']
        self.horizon = self.dataset_labeling_params['classifier_horizon']

        net_class_name = self.get_net_class()

        self.net = net_class_name(hparams=self.hparams, batch_size=batch_size, scenario=scenario)

        ckpt = tf.train.Checkpoint(model=self.net)
        manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=1)

        status = ckpt.restore(manager.latest_checkpoint).expect_partial()
        if manager.latest_checkpoint:
            print(Fore.CYAN + "Restored from {}".format(manager.latest_checkpoint) + Fore.RESET)
            if manager.latest_checkpoint:
                status.assert_nontrivial_match()
        else:
            raise RuntimeError(f"Failed to restore {manager.latest_checkpoint}!!!")

        self.state_keys = self.net.state_keys
        self.action_keys = self.net.action_keys
        self.true_state_keys = self.net.true_state_keys
        self.pred_state_keys = self.net.pred_state_keys

    # @tf.function
    def check_constraint_from_example(self, example: Dict, training: Optional[bool] = False):
        example_preprocessed = self.net.preprocess_no_gradient(example, training)
        predictions = self.net(example_preprocessed, training=training)
        return predictions

    def check_constraint_tf_batched(self,
                                    environment: Dict,
                                    states: Dict,
                                    actions: Dict,
                                    batch_size: int,
                                    state_sequence_length: int):
        # construct network inputs
        net_inputs = {
            'batch_size': batch_size,
            'time':       state_sequence_length,
        }
        if 'scene_msg' in environment:
            environment.pop('scene_msg')
        net_inputs.update(make_dict_tf_float32(environment))

        for action_key in self.action_keys:
            net_inputs[action_key] = tf.cast(actions[action_key], tf.float32)

        for state_key in self.state_keys:
            planned_state_key = add_predicted(state_key)
            net_inputs[planned_state_key] = tf.cast(states[state_key], tf.float32)

        if self.hparams['stdev']:
            net_inputs[add_predicted('stdev')] = tf.cast(states['stdev'], tf.float32)

        net_inputs = make_dict_tf_float32(net_inputs)
        predictions = self.check_constraint_from_example(net_inputs, training=False)
        probability = predictions['probabilities']
        probability = tf.squeeze(probability, axis=2)
        return probability

    def check_constraint_tf(self,
                            environment: Dict,
                            states_sequence: List[Dict],
                            actions: List[Dict]):
        environment = add_batch(environment)
        states_sequence_dict = sequence_of_dicts_to_dict_of_tensors(states_sequence)
        states_sequence_dict = add_batch(states_sequence_dict)
        state_sequence_length = len(states_sequence)
        actions_dict = sequence_of_dicts_to_dict_of_tensors(actions)
        actions_dict = add_batch(actions_dict)
        probabilities = self.check_constraint_tf_batched(environment=environment,
                                                         states=states_sequence_dict,
                                                         actions=actions_dict,
                                                         batch_size=1,
                                                         state_sequence_length=state_sequence_length)
        probabilities = remove_batch(probabilities)
        return probabilities

    def check_constraint(self,
                         environment: Dict,
                         states_sequence: List[Dict],
                         actions: List[Dict]):
        probabilities = self.check_constraint_tf(environment=environment,
                                                 states_sequence=states_sequence,
                                                 actions=actions)
        probabilities = probabilities.numpy()
        return probabilities

    @staticmethod
    def get_net_class():
        return NNClassifier
