#!/usr/bin/env python
from typing import Dict

import tensorflow as tf
from colorama import Fore
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy, Metric

import rospy
from jsk_recognition_msgs.msg import BoundingBox
from link_bot_data.dataset_utils import add_predicted
from link_bot_pycommon.bbox_visualization import grid_to_bbox
from link_bot_pycommon.grid_utils import batch_idx_to_point_3d_in_env_tf, \
    batch_point_to_idx_tf_3d_in_batched_envs, send_voxelgrid_tf, environment_to_vg_msg
from link_bot_pycommon.grid_utils import compute_extent_3d
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from merrrt_visualization.rviz_animation_controller import RvizSimpleStepper, RvizAnimationController
from moonshine.classifier_losses_and_metrics import class_weighted_mean_loss
from moonshine.get_local_environment import create_env_indices, get_local_env_and_origin_3d
from moonshine.metrics import BinaryAccuracyOnPositives, BinaryAccuracyOnNegatives, LossMetric, \
    FalsePositiveMistakeRate, \
    FalseNegativeMistakeRate, FalsePositiveOverallRate, FalseNegativeOverallRate
from moonshine.my_keras_model import MyKerasModel
from moonshine.raster_3d import raster_3d
from rviz_voxelgrid_visuals_msgs.msg import VoxelgridStamped

DEBUG_INPUT_ENV = False
DEBUG_PRE_AUG = False
DEBUG_POST_AUG = False
DEBUG_FITTED_ENV_AUG = False
DEBUG_ADDITIVE_AUG = False
SHOW_ALL = False


class NNClassifier(MyKerasModel):
    def __init__(self, hparams: Dict, batch_size: int, scenario: ScenarioWithVisualization):
        super().__init__(hparams, batch_size)
        self.scenario = scenario

        self.raster_debug_pubs = [
            rospy.Publisher(f'classifier_raster_debug_{i}', VoxelgridStamped, queue_size=10, latch=False) for i in
            range(5)]
        self.local_env_bbox_pub = rospy.Publisher('local_env_bbox', BoundingBox, queue_size=10, latch=True)
        self.env_aug_pub1 = rospy.Publisher("env_aug1", VoxelgridStamped, queue_size=10)
        self.env_aug_pub2 = rospy.Publisher("env_aug2", VoxelgridStamped, queue_size=10)
        self.env_aug_pub3 = rospy.Publisher("env_aug3", VoxelgridStamped, queue_size=10)
        self.env_aug_pub4 = rospy.Publisher("env_aug4", VoxelgridStamped, queue_size=10)
        self.env_aug_pub5 = rospy.Publisher("env_aug5", VoxelgridStamped, queue_size=10)

        self.classifier_dataset_hparams = self.hparams['classifier_dataset_hparams']
        self.dynamics_dataset_hparams = self.classifier_dataset_hparams['fwd_model_hparams']['dynamics_dataset_hparams']
        self.true_state_keys = self.classifier_dataset_hparams['true_state_keys']
        self.pred_state_keys = [add_predicted(k) for k in self.classifier_dataset_hparams['predicted_state_keys']]
        self.pred_state_keys.append(add_predicted('stdev'))
        self.local_env_h_rows = self.hparams['local_env_h_rows']
        self.local_env_w_cols = self.hparams['local_env_w_cols']
        self.local_env_c_channels = self.hparams['local_env_c_channels']
        self.rope_image_k = self.hparams['rope_image_k']

        self.env_augmentation_type = self.hparams.get('augmentation_type', None)
        if self.env_augmentation_type in ['env_augmentation_1', 'additive_env_resample_augmentation']:
            self.augmentation_3d = NNClassifier.additive_env_resample_augmentation
        elif self.env_augmentation_type in ['env_augmentation_2', 'fitted_env_augmentation']:
            self.augmentation_3d = NNClassifier.fitted_env_augmentation
        else:
            # NOTE: this gets hacked in fine_tune_classifier.py when you pass in "augmentation_3d=my_aug_func"
            self.augmentation_3d = None

        self.aug_valid_type = self.hparams.get('augmentation_valid_type', None)
        if self.aug_valid_type in ['discrete']:
            self.is_env_augmentation_valid = NNClassifier.is_env_augmentation_valid_discrete
        elif self.aug_valid_type in ['swept']:
            self.is_env_augmentation_valid = NNClassifier.is_env_augmentation_valid_swept
        else:
            self.is_env_augmentation_valid = None

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

        self.aug_generator = tf.random.Generator.from_seed(0)

    # @tf.function
    def call(self, input_dict: Dict, training, **kwargs):
        batch_size = input_dict['batch_size']
        time = tf.cast(input_dict['time'], tf.int32)

        if DEBUG_INPUT_ENV:
            stepper = RvizSimpleStepper()
            for b in self.debug_viz_batch_indices():
                env_b = {
                    'env':    input_dict['env'][b],
                    'res':    input_dict['res'][b],
                    'extent': input_dict['extent'][b],
                    'origin': input_dict['origin'][b],
                }
                self.scenario.plot_environment_rviz(env_b)
                stepper.step()

        voxel_grids = self.make_voxel_grid_inputs(input_dict, batch_size=batch_size, time=time, training=training)

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

    def make_voxel_grid_inputs(self, input_dict: Dict, batch_size, time, training: bool):
        indices = self.create_env_indices(batch_size)

        local_env, local_env_origin = self.get_local_env(batch_size, indices, input_dict)

        local_voxel_grids_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        for t in tf.range(time):
            state_t = {k: input_dict[add_predicted(k)][:, t] for k in self.state_keys}

            local_voxel_grid_t_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
            # Insert the environment as channel 0
            local_voxel_grid_t_array = local_voxel_grid_t_array.write(0, local_env)

            # insert the rastered states
            for i, (k, state_component_t) in enumerate(state_t.items()):
                state_component_voxel_grid = raster_3d(state=state_component_t,
                                                       pixel_indices=indices.pixels,
                                                       res=input_dict['res'],
                                                       origin=local_env_origin,
                                                       h=self.local_env_h_rows,
                                                       w=self.local_env_w_cols,
                                                       c=self.local_env_c_channels,
                                                       k=self.rope_image_k,
                                                       batch_size=batch_size)

                local_voxel_grid_t_array = local_voxel_grid_t_array.write(i + 1, state_component_voxel_grid)
            local_voxel_grid_t = tf.transpose(local_voxel_grid_t_array.stack(), [1, 2, 3, 4, 0])
            # add channel dimension information because tf.function erases it somehow...
            local_voxel_grid_t.set_shape([None, None, None, None, len(self.state_keys) + 1])

            local_voxel_grids_array = local_voxel_grids_array.write(t, local_voxel_grid_t)

        if DEBUG_PRE_AUG:
            self.debug_viz_local_env_pre_aug(input_dict, local_voxel_grids_array, local_env_origin, time)

        # optionally augment the local environment
        if self.augmentation_3d is not None and training:
            local_voxel_grids_aug_array = self.augmentation_3d(self, time, input_dict, local_voxel_grids_array, indices)
        else:
            local_voxel_grids_aug_array = local_voxel_grids_array

        if DEBUG_POST_AUG:
            stepper = RvizSimpleStepper()
            for b in self.debug_viz_batch_indices():
                final_aug_dict = {
                    'env':    local_voxel_grids_aug_array.read(0)[b, :, :, :, 0].numpy(),
                    'origin': local_env_origin[b].numpy(),
                    'res':    input_dict['res'][b].numpy(),
                }
                msg = environment_to_vg_msg(final_aug_dict, frame='local_env', stamp=rospy.Time(0))
                self.env_aug_pub5.publish(msg)
                state_0 = {k: input_dict[add_predicted(k)][:, 0] for k in self.state_keys}
                action_0 = {k: input_dict[k][:, 0] for k in self.action_keys}
                state_1 = {k: input_dict[add_predicted(k)][:, 1] for k in self.state_keys}
                self.scenario.plot_state_rviz(state_0, idx=0)
                self.scenario.plot_state_rviz(state_1, idx=1)
                self.scenario.plot_action_rviz(state_0, action_0, idx=1)
                stepper.step()

            for b in self.debug_viz_batch_indices():
                self.animate_voxel_grid_states(b, input_dict, local_env_origin, local_voxel_grids_array, time)

        return local_voxel_grids_aug_array

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
        # by converting too and from the frame of the full environment, we ensure the grids are aligned
        center_indices = batch_point_to_idx_tf_3d_in_batched_envs(local_env_center, input_dict)
        local_env_center = batch_idx_to_point_3d_in_env_tf(*center_indices, input_dict)
        # local_env, local_env_origin = get_local_env_and_origin(center_point=local_env_center,
        #                                                        full_env=input_dict['env'],
        #                                                        full_env_origin=input_dict['origin'],
        #                                                        res=input_dict['res'],
        #                                                        local_h_rows=self.local_env_h_rows,
        #                                                        local_w_cols=self.local_env_w_cols,
        #                                                        local_c_channels=self.local_env_c_channels,
        #                                                        batch_x_indices=indices.batch_x,
        #                                                        batch_y_indices=indices.batch_y,
        #                                                        batch_z_indices=indices.batch_z,
        #                                                        batch_size=batch_size)
        local_env, local_env_origin = self.get_new_local_env()
        return local_env, local_env_origin

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

    def local_env_given_center(self, center_point, environment: Dict):
        return get_local_env_and_origin_3d(center_point=center_point,
                                           environment=environment,
                                           h=self.local_env_h_rows,
                                           w=self.local_env_w_cols,
                                           c=self.local_env_c_channels,
                                           indices=self.indices,
                                           batch_size=self.batch_size)

    def debug_viz_local_env_pre_aug(self, example: Dict, local_voxel_grids_array, local_env_origin, time):
        for b in self.debug_viz_batch_indices():
            bbox_msg = grid_to_bbox(rows=self.local_env_h_rows,
                                    cols=self.local_env_w_cols,
                                    channels=self.local_env_c_channels,
                                    resolution=example['res'][b].numpy())
            bbox_msg.header.frame_id = 'local_env'
            self.local_env_bbox_pub.publish(bbox_msg)

            local_env_extent_b = compute_extent_3d(self.local_env_h_rows,
                                                   self.local_env_w_cols,
                                                   self.local_env_c_channels,
                                                   example['res'][b],
                                                   local_env_origin[b])
            local_env_dict = {
                'extent': local_env_extent_b,
                'origin': local_env_origin[b].numpy(),
                'res':    example['res'][b].numpy(),
            }
            send_voxelgrid_tf(self.scenario.tf.tf_broadcaster, local_env_dict, frame='local_env')

            self.animate_voxel_grid_states(b, example, local_env_origin, local_voxel_grids_array, time)

    def animate_voxel_grid_states(self, b, example, local_env_origin, local_voxel_grids_array, time):
        anim = RvizAnimationController(n_time_steps=time)
        while not anim.done:
            t = anim.t()

            local_voxel_grid_t = local_voxel_grids_array.read(t)

            for i, state_component_k_voxel_grid in enumerate(tf.transpose(local_voxel_grid_t, [4, 0, 1, 2, 3])):
                raster_dict = {
                    'env':    tf.clip_by_value(state_component_k_voxel_grid[b], 0, 1),
                    'origin': local_env_origin[b].numpy(),
                    'res':    example['res'][b].numpy(),
                }
                raster_msg = environment_to_vg_msg(raster_dict, frame='local_env', stamp=rospy.Time(0))
                self.raster_debug_pubs[i].publish(raster_msg)

            anim.step()

    def debug_viz_batch_indices(self):
        if SHOW_ALL:
            return range(self.batch_size)
        else:
            return [8]