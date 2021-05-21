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
from jsk_recognition_msgs.msg import BoundingBox
from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_data.dataset_utils import add_predicted, add_new
from link_bot_pycommon.bbox_visualization import grid_to_bbox
from link_bot_pycommon.debugging_utils import debug_viz_batch_indices
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.grid_utils import batch_extent_to_origin_point_tf, environment_to_vg_msg, \
    vox_to_voxelgrid_stamped, send_voxelgrid_tf_origin_point_res_tf, occupied_voxels_to_points, binary_or, binary_and, \
    subtract
from link_bot_pycommon.grid_utils import batch_point_to_idx
from link_bot_pycommon.pycommon import make_dict_tf_float32
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from merrrt_visualization.rviz_animation_controller import RvizSimpleStepper, RvizAnimationController
from moonshine.classifier_losses_and_metrics import class_weighted_mean_loss
from moonshine.geometry import make_rotation_matrix_like, rotate_points_3d, pairwise_squared_distances
from moonshine.get_local_environment import create_env_indices, get_local_env_and_origin_3d
from moonshine.metrics import BinaryAccuracyOnPositives, BinaryAccuracyOnNegatives, LossMetric, \
    FalsePositiveMistakeRate, FalseNegativeMistakeRate, FalsePositiveOverallRate, FalseNegativeOverallRate
from moonshine.moonshine_utils import add_batch, remove_batch, sequence_of_dicts_to_dict_of_tensors, numpify, \
    to_list_of_strings
from moonshine.my_keras_model import MyKerasModel
from moonshine.optimization import log_barrier
from moonshine.raster_3d import batch_points_to_voxel_grid_res_origin_point, points_to_voxel_grid_res_origin_point
from moveit_msgs.msg import RobotTrajectory
from rviz_voxelgrid_visuals_msgs.msg import VoxelgridStamped
from std_msgs.msg import Float32
from trajectory_msgs.msg import JointTrajectoryPoint
from visualization_msgs.msg import MarkerArray, Marker

DEBUG_INPUT = False
DEBUG_AUG = False
DEBUG_AUG_SGD = False
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
        self.local_env_new_bbox_pub = rospy.Publisher('local_env_new_bbox', BoundingBox, queue_size=10, latch=True)
        self.aug_bbox_pub = rospy.Publisher('local_env_bbox_aug', BoundingBox, queue_size=10, latch=True)
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

        self.aug_hparams = self.hparams.get('augmentation', None)
        # if self.aug_hparams.get('swept', True):

        self.aug_gen = tf.random.Generator.from_seed(0)
        self.aug_seed_stream = tfp.util.SeedStream(1, salt="nn_classifier_aug")
        self.aug_opt = tf.keras.optimizers.SGD(0.1)
        self.aug_opt_grad_norm_threshold = 0.008  # stopping criteria for the eng aug optimization
        self.barrier_upper_cutoff = tf.square(0.04)  # stops repelling points from pushing after this distance
        self.barrier_scale = 1.1  # scales the gradients for the repelling points
        self.env_aug_grad_clip = 5.0  # max dist step the env aug update can take

    def preprocess_no_gradient(self, example, training: bool):
        example['origin_point'] = batch_extent_to_origin_point_tf(example['extent'], example['res'])
        return example

    def call(self, input_dict: Dict, training, **kwargs):
        batch_size = input_dict['batch_size']
        time = tf.cast(input_dict['time'], tf.int32)

        if DEBUG_INPUT:
            # clear the other voxel grids from previous calls
            vg_empty = np.zeros((64, 64, 64))
            empty_msg = vox_to_voxelgrid_stamped(vg_empty, scale=0.01, frame='world')
            for p in self.raster_debug_pubs:
                p.publish(empty_msg)

            self.scenario.delete_points_rviz(label='attract')
            self.scenario.delete_points_rviz(label='repel')
            self.scenario.delete_points_rviz(label='attract_aug')
            self.scenario.delete_points_rviz(label='repel_aug')
            self.scenario.delete_lines_rviz(label='attract')
            self.scenario.delete_lines_rviz(label='repel')
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
                stepper.step()

        # Create voxel grids
        indices = self.create_env_indices(batch_size)
        local_env, local_origin_point = self.get_local_env(indices, input_dict)

        voxel_grids = self.make_voxelgrid_inputs(input_dict, local_env, local_origin_point, batch_size, time)

        if DEBUG_AUG:
            self.debug_viz_local_env_pre_aug(input_dict, voxel_grids, local_origin_point, time)

        if training and self.aug_hparams is not None:
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

    # @tf.function
    def conv_encoder(self, voxel_grids, batch_size, time):
        conv_outputs_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for t in range(time):
            conv_z = voxel_grids[:, t]
            for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
                conv_h = conv_layer(conv_z)
                conv_z = pool_layer(conv_h)
            out_conv_z = conv_z
            out_conv_z_dim = out_conv_z.shape[1] * out_conv_z.shape[2] * out_conv_z.shape[3] * out_conv_z.shape[4]
            out_conv_z = tf.reshape(out_conv_z, [batch_size, out_conv_z_dim])
            conv_outputs_array = conv_outputs_array.write(t, out_conv_z)
        conv_outputs = conv_outputs_array.stack()
        conv_outputs = tf.transpose(conv_outputs, [1, 0, 2])
        return conv_outputs

    # @tf.function
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

    # @tf.function
    def create_env_indices(self, batch_size: int):
        return create_env_indices(self.local_env_h_rows, self.local_env_w_cols, self.local_env_c_channels, batch_size)

    # @tf.function
    def make_voxelgrid_inputs(self, input_dict: Dict, local_env, local_origin_point, batch_size, time):
        local_voxel_grids_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        for t in tf.range(time):
            state_t = {k: input_dict[add_predicted(k)][:, t] for k in self.state_keys}

            local_voxel_grid_t_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
            # Insert the environment as channel 0
            local_voxel_grid_t_array = local_voxel_grid_t_array.write(0, local_env)

            # insert the rastered states
            channel_idx = 0
            for channel_idx, (k, state_component_t) in enumerate(state_t.items()):
                n_points_in_component = int(state_component_t.shape[1] / 3)
                points = tf.reshape(state_component_t, [batch_size, -1, 3])
                flat_batch_indices = tf.repeat(tf.range(batch_size, dtype=tf.int64), n_points_in_component, axis=0)
                flat_points = tf.reshape(points, [-1, 3])
                flat_points.set_shape([n_points_in_component * self.batch_size, 3])
                flat_res = tf.repeat(input_dict['res'], n_points_in_component, axis=0)
                flat_origin_point = tf.repeat(local_origin_point, n_points_in_component, axis=0)
                state_component_voxel_grid = batch_points_to_voxel_grid_res_origin_point(flat_batch_indices,
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
            # add channel dimension information because tf.function erases it?
            local_voxel_grid_t.set_shape([None, None, None, None, n_channels])

            local_voxel_grids_array = local_voxel_grids_array.write(t, local_voxel_grid_t)

        local_voxel_grids = tf.transpose(local_voxel_grids_array.stack(), [1, 0, 2, 3, 4, 5])
        local_voxel_grids.set_shape([None, 2, None, None, None, None])  # FIXME: 2 is hardcoded here
        return local_voxel_grids

    def get_local_env(self, indices, input_dict):
        state_0 = {k: input_dict[add_predicted(k)][:, 0] for k in self.state_keys}

        # NOTE: to be more general, this should return a pose not just a point/position
        local_env_center = self.scenario.local_environment_center_differentiable(state_0)
        environment = {k: input_dict[k] for k in ['env', 'origin_point', 'res', 'extent']}
        local_env, local_origin_point = self.local_env_given_center(local_env_center, environment, indices)

        if DEBUG_AUG:
            stepper = RvizSimpleStepper()
            for b in debug_viz_batch_indices(self.batch_size):
                self.scenario.tf.send_transform(local_env_center[b], [0, 0, 0, 1], 'world', 'local_env_center',
                                                is_static=True)
                self.scenario.tf.send_transform(local_origin_point[b], [0, 0, 0, 1], 'world', 'local_origin_point',
                                                is_static=True)
                # stepper.step()

        return local_env, local_origin_point

    def merge_aug_and_local_voxel_grids(self, env, voxel_grids, time):
        env_with_time = tf.tile(env[:, None, :, :, :, None], [1, time, 1, 1, 1, 1])
        voxel_grids_without_env = voxel_grids[:, :, :, :, :, 1:]
        voxel_grids = tf.concat([env_with_time, voxel_grids_without_env], axis=-1)
        voxel_grids = tf.clip_by_value(voxel_grids, 0.0, 1.0)
        return voxel_grids

    def get_new_env(self, example):
        if add_new('env') not in example:
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
        return new_env

    def sample_local_env_position(self, example):
        # NOTE: for my specific implementation of state_to_local_env_pose,
        #  sampling random states and calling state_to_local_env_pose is equivalent to sampling a point in the extent
        extent = tf.reshape(example['extent'], [self.batch_size, 3, 2])
        extent_lower = tf.gather(extent, 0, axis=-1)
        extent_upper = tf.gather(extent, 1, axis=-1)
        local_env_center = self.aug_gen.uniform([self.batch_size, 3], extent_lower, extent_upper)

        return local_env_center

    def local_env_given_center(self, center_point, environment: Dict, indices: Dict):
        return get_local_env_and_origin_3d(center_point=center_point,
                                           environment=environment,
                                           h=self.local_env_h_rows,
                                           w=self.local_env_w_cols,
                                           c=self.local_env_c_channels,
                                           indices=indices,
                                           batch_size=self.batch_size)

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

    def augmentation_optimization(self,
                                  example: Dict,
                                  indices: Dict,
                                  local_origin_point,
                                  voxel_grids,
                                  batch_size,
                                  time):
        # before augmentation, get all components of the state as a set of points
        # in general this should be the swept volume, and should include the robot
        states = {k: example[add_predicted(k)] for k in self.state_keys}
        state_points = tf.concat([tf.reshape(v, [batch_size, time, -1, 3]) for v in states.values()], axis=2)
        state_points = tf.reshape(state_points, [batch_size, -1, 3])
        res = example['res']

        # sample a translation and rotation for the object state
        translation, theta = self.scenario.sample_state_augmentation_variables(batch_size, self.aug_seed_stream)
        rotation = make_rotation_matrix_like(translation, theta)

        valid, local_origin_point_aug = self.scenario.apply_state_augmentation(translation,
                                                                               rotation,
                                                                               example,
                                                                               local_origin_point,
                                                                               batch_size,
                                                                               time,
                                                                               self.local_env_h_rows,
                                                                               self.local_env_w_cols,
                                                                               self.local_env_c_channels)

        local_env = voxel_grids[:, 0, :, :, :, 0]  # just use 0 because it's the same at all time steps
        local_env_occupancy = self.lookup_points_in_vg(state_points, local_env, res, local_origin_point, batch_size)

        if DEBUG_AUG:
            stepper = RvizSimpleStepper()
            for b in debug_viz_batch_indices(batch_size):
                debug_i = tf.squeeze(tf.where(1 - local_env_occupancy[b]), -1)
                points_debug_b = tf.gather(state_points[b], debug_i)
                self.scenario.plot_points_rviz(points_debug_b.numpy(), label='repel', color='r')

                debug_i = tf.squeeze(tf.where(local_env_occupancy[b]), -1)
                points_debug_b = tf.gather(state_points[b], debug_i)
                self.scenario.plot_points_rviz(points_debug_b.numpy(), label='attract', color='g')
                # stepper.step()

                send_voxelgrid_tf_origin_point_res_tf(self.broadcaster,
                                                      origin_point=local_origin_point_aug[b],
                                                      res=res[b],
                                                      frame='local_env_aug_vg')

                bbox_msg = grid_to_bbox(rows=self.local_env_h_rows,
                                        cols=self.local_env_w_cols,
                                        channels=self.local_env_c_channels,
                                        resolution=res[b].numpy())
                bbox_msg.header.frame_id = 'local_env_aug_vg'

                self.aug_bbox_pub.publish(bbox_msg)

        state_points_aug = rotate_points_3d(rotation[:, None], state_points) + translation[:, None]
        valid_expanded = valid[:, None, None]
        state_points_aug = valid_expanded * state_points_aug + (1 - valid_expanded) * state_points

        if DEBUG_AUG:
            for b in debug_viz_batch_indices(batch_size):
                debug_i = tf.squeeze(tf.where(local_env_occupancy[b]), -1)
                points_debug_b = tf.gather(state_points_aug[b], debug_i)
                self.scenario.plot_points_rviz(points_debug_b.numpy(), label='attract_aug', color='g')

                debug_i = tf.squeeze(tf.where(1 - local_env_occupancy[b]), -1)
                points_debug_b = tf.gather(state_points_aug[b], debug_i)
                self.scenario.plot_points_rviz(points_debug_b.numpy(), label='repel_aug', color='r')

        new_env = self.get_new_env(example)
        local_env_aug = self.opt_new_env_augmentation(new_env,
                                                      indices,
                                                      state_points_aug,
                                                      local_env_occupancy,
                                                      res,
                                                      local_origin_point_aug,
                                                      batch_size)

        if DEBUG_AUG:
            stepper = RvizSimpleStepper()
            for b in debug_viz_batch_indices(batch_size):
                _aug_dict = {
                    'env':          local_env_aug[b].numpy(),
                    'origin_point': local_origin_point_aug[b].numpy(),
                    'res':          res[b].numpy(),
                }
                msg = environment_to_vg_msg(_aug_dict, frame='local_env_aug_vg', stamp=rospy.Time(0))
                self.env_aug_pub5.publish(msg)
                send_voxelgrid_tf_origin_point_res_tf(self.broadcaster,
                                                      local_origin_point_aug[b],
                                                      res[b],
                                                      frame='local_env_aug_vg')

                self.debug_viz_state_action(example, b, 'aug', color='blue')
                stepper.step()

        voxel_grids_aug = self.merge_aug_and_local_voxel_grids(local_env_aug,
                                                               voxel_grids,
                                                               time)
        return voxel_grids_aug

    def opt_new_env_augmentation(self,
                                 new_env: Dict,
                                 indices: Dict,
                                 state_points_aug,
                                 local_env_occupancy,
                                 res,
                                 local_origin_point_aug,
                                 batch_size):
        local_env_new_center = self.sample_local_env_position(new_env)
        local_env_new, local_env_new_origin_point = self.local_env_given_center(local_env_new_center, new_env, indices)
        # viz new env
        if DEBUG_AUG:
            for b in debug_viz_batch_indices(self.batch_size):
                self.scenario.tf.send_transform(local_env_new_center[b], [0, 0, 0, 1], 'world', 'local_env_new_center',
                                                is_static=True)

                send_voxelgrid_tf_origin_point_res_tf(self.broadcaster,
                                                      origin_point=local_env_new_origin_point[b],
                                                      res=res[b],
                                                      frame='local_env_new_vg')

                bbox_msg = grid_to_bbox(rows=self.local_env_h_rows,
                                        cols=self.local_env_w_cols,
                                        channels=self.local_env_c_channels,
                                        resolution=res[b].numpy())
                bbox_msg.header.frame_id = 'local_env_new_vg'

                self.local_env_new_bbox_pub.publish(bbox_msg)

                env_new_dict = {
                    'env': new_env['env'][b].numpy(),
                    'res': res[b].numpy(),
                }
                msg = environment_to_vg_msg(env_new_dict, frame='new_env_aug_vg', stamp=rospy.Time(0))
                self.env_aug_pub1.publish(msg)

                send_voxelgrid_tf_origin_point_res_tf(self.broadcaster,
                                                      origin_point=new_env['origin_point'][b],
                                                      res=res[b],
                                                      frame='new_env_aug_vg')

                # Show sample new local environment, in the frame of the original local env, the one we're augmenting
                local_env_new_dict = {
                    'env': local_env_new[b].numpy(),
                    'res': res[b].numpy(),
                }
                msg = environment_to_vg_msg(local_env_new_dict, frame='local_env_aug_vg', stamp=rospy.Time(0))
                self.env_aug_pub2.publish(msg)

                send_voxelgrid_tf_origin_point_res_tf(self.broadcaster,
                                                      origin_point=local_origin_point_aug[b],
                                                      res=res[b],
                                                      frame='local_env_aug_vg')

                # stepper.step()

        n_steps = 100
        if DEBUG_AUG_SGD:
            stepper = RvizSimpleStepper()

        nearest_attract_points = None
        nearest_repel_points = None
        attract_points_b = None
        repel_points_b = None
        local_env_aug = []
        for b in range(batch_size):
            r_b = res[b]
            o_b = local_origin_point_aug[b]
            state_points_b = state_points_aug[b]
            local_env_occupancy_b = local_env_occupancy[b]
            env_points_b_initial = occupied_voxels_to_points(local_env_new[b], r_b, o_b)
            env_points_b = env_points_b_initial

            translation_b = tf.Variable([0, 0, 0], dtype=tf.float32)
            variables = [translation_b]
            for i in range(n_steps):
                with tf.GradientTape() as tape:
                    env_points_b = env_points_b_initial + translation_b
                    is_attract_indices = tf.squeeze(tf.where(local_env_occupancy_b > 0.5), 1)
                    attract_points_b = tf.gather(state_points_b, is_attract_indices)
                    if tf.size(is_attract_indices) == 0:
                        attract_loss = 0
                    else:
                        attract_dists_b = pairwise_squared_distances(env_points_b, attract_points_b)
                        min_attract_dist_indices_b = tf.argmin(attract_dists_b, axis=1)
                        min_attract_dist_b = tf.reduce_min(attract_dists_b, axis=1)
                        nearest_attract_points = tf.gather(attract_points_b, min_attract_dist_indices_b)
                        attract_loss = tf.reduce_mean(min_attract_dist_b)

                    is_repel_indices = tf.squeeze(tf.where(local_env_occupancy_b < 0.5), 1)
                    repel_points_b = tf.gather(state_points_b, is_repel_indices)
                    if tf.size(is_repel_indices) == 0:
                        repel_loss = 0
                    else:
                        repel_dists_b = pairwise_squared_distances(env_points_b, repel_points_b)
                        min_repel_dist_indices_b = tf.argmin(repel_dists_b, axis=1)
                        min_repel_dist_b = tf.reduce_min(repel_dists_b, axis=1)
                        nearest_repel_points = tf.gather(repel_points_b, min_repel_dist_indices_b)
                        repel_loss = tf.reduce_mean(self.barrier_func(min_repel_dist_b))

                    loss = attract_loss + repel_loss

                if DEBUG_AUG_SGD:
                    if b in debug_viz_batch_indices(batch_size):
                        self.scenario.plot_points_rviz(env_points_b, label='icp', color='grey')
                        self.scenario.plot_lines_rviz(nearest_attract_points, env_points_b, label='attract', color='g')
                        self.scenario.plot_lines_rviz(nearest_repel_points, env_points_b, label='repel', color='r')
                        # stepper.step()

                gradients = tape.gradient(loss, variables)

                clipped_grads_and_vars = [(self.clip_env_aug_grad(g), v) for (g, v) in zip(gradients, variables)]
                self.aug_opt.apply_gradients(grads_and_vars=clipped_grads_and_vars)

                grad_norm = tf.linalg.norm(gradients)
                if grad_norm < self.aug_opt_grad_norm_threshold:
                    break
            local_env_aug_b = self.points_to_voxel_grid_res_origin_point(env_points_b, r_b, o_b)

            # after local optimization, enforce the constraint by ensuring voxels with attract points are on,
            # and voxels with repel points are off
            attract_vg_b = self.points_to_voxel_grid_res_origin_point(attract_points_b, r_b, o_b)
            repel_vg_b = self.points_to_voxel_grid_res_origin_point(repel_points_b, r_b, o_b)
            local_env_aug_b = subtract(binary_or(local_env_aug_b, attract_vg_b), repel_vg_b)

            local_env_aug.append(local_env_aug_b)

        local_env_aug = tf.stack(local_env_aug)

        return local_env_aug

    def points_to_voxel_grid_res_origin_point(self, points, res, origin_point):
        return points_to_voxel_grid_res_origin_point(points,
                                                     res,
                                                     origin_point,
                                                     self.local_env_h_rows,
                                                     self.local_env_w_cols,
                                                     self.local_env_c_channels)

    def clip_env_aug_grad(self, grad):
        return tf.clip_by_value(grad, -self.env_aug_grad_clip, self.env_aug_grad_clip)

    def barrier_func(self, min_dists_b):
        return log_barrier(min_dists_b, scale=self.barrier_scale, cutoff=self.barrier_upper_cutoff)

    def lookup_points_in_vg(self, state_points, local_env, res, local_origin_point, batch_size):
        """
        Returns the values of local_env at state_points
        Args:
            state_points: [b, n, 3], in same frame as local_origin_point
            local_env: [b, h, w, c]
            res:
            local_origin_point: [b, 3] in same frame as state_points
            batch_size:

        Returns: [b, n]

        """
        n_points = state_points.shape[1]
        vg_indices = batch_point_to_idx(state_points,
                                        tf.expand_dims(res, axis=1),
                                        tf.expand_dims(local_origin_point, axis=1))
        batch_indices = tf.tile(tf.range(batch_size)[:, None, None], [1, n_points, 1])
        batch_and_vg_indices = tf.concat([batch_indices, vg_indices], axis=-1)
        occupancy_at_state_points = tf.gather_nd(local_env, batch_and_vg_indices)  # [b, n_points, 2]
        return occupancy_at_state_points

    def debug_viz_local_env_pre_aug(self, example: Dict, voxel_grids, local_origin_point, time):
        for b in debug_viz_batch_indices(self.batch_size):
            send_voxelgrid_tf_origin_point_res_tf(self.broadcaster,
                                                  origin_point=local_origin_point[b],
                                                  res=example['res'][b],
                                                  frame='local_env_vg')

            bbox_msg = grid_to_bbox(rows=self.local_env_h_rows,
                                    cols=self.local_env_w_cols,
                                    channels=self.local_env_c_channels,
                                    resolution=example['res'][b].numpy())
            bbox_msg.header.frame_id = 'local_env_vg'

            self.local_env_bbox_pub.publish(bbox_msg)

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
        state_keys = ['left_gripper', 'right_gripper', 'rope', 'joint_positions']
        state_0 = numpify({k: input_dict[add_predicted(k)][b, 0] for k in state_keys})
        state_0['joint_names'] = input_dict['joint_names'][b, 0]
        action_0 = numpify({k: input_dict[k][b, 0] for k in self.action_keys})
        state_1 = numpify({k: input_dict[add_predicted(k)][b, 1] for k in state_keys})
        state_1['joint_names'] = input_dict['joint_names'][b, 1]
        error_msg = Float32()
        error_t = input_dict['error'][b, 1]
        error_msg.data = error_t
        self.scenario.plot_state_rviz(state_0, idx=0, label=label, color=color)
        self.scenario.plot_state_rviz(state_1, idx=1, label=label, color=color)
        robot_state = {k: input_dict[k][b] for k in ['joint_names', add_predicted('joint_positions')]}
        display_traj_msg = self.make_robot_trajectory(robot_state)
        self.scenario.robot.display_robot_traj(display_traj_msg, label=label, color=color)
        self.scenario.plot_action_rviz(state_0, action_0, idx=1, label=label, color=color)
        self.scenario.plot_is_close(input_dict['is_close'][b, 1])
        self.scenario.error_pub.publish(error_msg)

    def make_robot_trajectory(self, robot_state: Dict):
        msg = RobotTrajectory()
        # use 0 because joint names will be the same at every time step anyways
        msg.joint_trajectory.joint_names = to_list_of_strings(robot_state['joint_names'][0])
        for i, position in enumerate(robot_state[add_predicted('joint_positions')]):
            point = JointTrajectoryPoint()
            point.positions = numpify(position)
            point.time_from_start.secs = i  # not really "time" but that's fine, it's just for visualization
            msg.joint_trajectory.points.append(point)
        return msg


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
