from copy import copy
from typing import Dict

import tensorflow as tf
import transformations
from colorama import Fore
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy, Metric

import rospy
from link_bot_classifiers.classifier_augmentation import ClassifierAugmentation
from link_bot_classifiers.classifier_debugging import ClassifierDebugging
from link_bot_classifiers.make_voxelgrid_inputs import make_voxelgrid_inputs_t, VoxelgridInfo
from link_bot_classifiers.robot_points import RobotVoxelgridInfo
from link_bot_data.dataset_utils import add_predicted, add_new
from link_bot_pycommon.bbox_visualization import grid_to_bbox
from link_bot_pycommon.debugging_utils import debug_viz_batch_indices
from link_bot_pycommon.grid_utils import batch_extent_to_origin_point_tf, environment_to_vg_msg, \
    send_voxelgrid_tf_origin_point_res, occupied_voxels_to_points, binary_or, subtract, lookup_points_in_vg
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from merrrt_visualization.rviz_animation_controller import RvizSimpleStepper, RvizAnimationController
from moonshine.classifier_losses_and_metrics import class_weighted_mean_loss
from moonshine.geometry import pairwise_squared_distances, transform_points_3d
from moonshine.get_local_environment import create_env_indices, get_local_env_and_origin_point
from moonshine.metrics import BinaryAccuracyOnPositives, BinaryAccuracyOnNegatives, LossMetric, \
    FalsePositiveMistakeRate, FalseNegativeMistakeRate, FalsePositiveOverallRate, FalseNegativeOverallRate
from moonshine.moonshine_utils import numpify, to_list_of_strings
from moonshine.my_keras_model import MyKerasModel
from moonshine.optimization import log_barrier
from moonshine.raster_3d import points_to_voxel_grid_res_origin_point
from moveit_msgs.msg import RobotTrajectory
from std_msgs.msg import Float32
from trajectory_msgs.msg import JointTrajectoryPoint
from visualization_msgs.msg import MarkerArray, Marker

DEBUG_INPUT = False
DEBUG_AUG = False
DEBUG_AUG_SGD = False


class NNClassifier(MyKerasModel):
    def __init__(self, hparams: Dict, batch_size: int, scenario: ScenarioWithVisualization):
        super().__init__(hparams, batch_size)
        self.scenario = scenario
        self.broadcaster = self.scenario.tf.tf_broadcaster

        # define network structure from hparams
        self.classifier_dataset_hparams = self.hparams['classifier_dataset_hparams']
        self.dynamics_dataset_hparams = self.classifier_dataset_hparams['fwd_model_hparams']['dynamics_dataset_hparams']
        self.true_state_keys = self.classifier_dataset_hparams['true_state_keys']
        self.pred_state_keys = [add_predicted(k) for k in self.classifier_dataset_hparams['predicted_state_keys']]
        self.pred_state_keys.append(add_predicted('stdev'))
        self.local_env_h_rows = self.hparams['local_env_h_rows']
        self.local_env_w_cols = self.hparams['local_env_w_cols']
        self.local_env_c_channels = self.hparams['local_env_c_channels']
        self.state_keys = self.hparams['state_keys']
        self.points_state_keys = copy(self.state_keys)
        self.points_state_keys.remove("joint_positions")  # FIXME: feels hacky
        self.state_metadata_keys = self.hparams['state_metadata_keys']
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

        self.debug = ClassifierDebugging()
        self.aug = ClassifierAugmentation(self.hparams, self.batch_size, self.scenario)
        if self.aug.do_augmentation():
            rospy.loginfo("Using augmentation during training")
        else:
            rospy.loginfo("Not using augmentation during training")

        self.indices = self.create_env_indices(batch_size)
        self.include_robot_geometry = self.hparams.get('include_robot_geometry', False)
        print(Fore.LIGHTBLUE_EX + f"{self.include_robot_geometry=}" + Fore.RESET)
        self.robot_info = RobotVoxelgridInfo(joint_positions_key=add_predicted('joint_positions'))

        self.vg_info = VoxelgridInfo(h=self.local_env_h_rows,
                                     w=self.local_env_w_cols,
                                     c=self.local_env_c_channels,
                                     state_keys=[add_predicted(k) for k in self.points_state_keys],
                                     jacobian_follower=self.scenario.robot.jacobian_follower,
                                     robot_info=self.robot_info,
                                     )

    def preprocess_no_gradient(self, inputs, training: bool):
        batch_size = inputs['batch_size']
        time = inputs['time']

        inputs['origin_point'] = batch_extent_to_origin_point_tf(inputs['extent'], inputs['res'])

        if DEBUG_INPUT:
            # clear the other voxel grids from previous calls
            self.debug.clear()
            self.scenario.delete_points_rviz(label='attract')
            self.scenario.delete_points_rviz(label='repel')
            self.scenario.delete_points_rviz(label='attract_aug')
            self.scenario.delete_points_rviz(label='repel_aug')
            self.scenario.delete_lines_rviz(label='attract')
            self.scenario.delete_lines_rviz(label='repel')

            stepper = RvizSimpleStepper()
            for b in debug_viz_batch_indices(batch_size):
                env_b = {
                    'env':          inputs['env'][b],
                    'res':          inputs['res'][b],
                    'origin_point': inputs['origin_point'][b],
                    'extent':       inputs['extent'][b],
                }
                self.scenario.plot_environment_rviz(env_b)
                self.delete_state_action_markers('aug')
                self.debug_viz_state_action(inputs, b, 'input')
                origin_point_b = inputs['origin_point'][b].numpy().tolist()
                self.send_position_transform(origin_point_b, 'env_origin_point')
                # stepper.step()

        # Create voxel grids
        local_env, local_origin_point = self.get_local_env(inputs, batch_size)

        voxel_grids = self.make_voxelgrid_inputs(inputs, local_env, local_origin_point, batch_size, time)

        inputs['voxel_grids'] = voxel_grids
        inputs['local_origin_point'] = local_origin_point

        inputs['swept_state_and_robot_points'] = self.scenario.compute_swept_state_and_robot_points(inputs)

        if DEBUG_AUG:
            self.debug_viz_local_env_pre_aug(inputs, time)

        if training and self.aug.do_augmentation():
            # input_dict is also modified, but in place because it's a dict, where as voxel_grids is a tensor and
            # so modifying it internally won't change the value for the caller
            inputs['voxel_grids'] = self.augmentation_optimization(inputs, batch_size, time)

        return inputs

    def call(self, inputs: Dict, training, **kwargs):
        batch_size = inputs['batch_size']
        time = tf.cast(inputs['time'], tf.int32)
        voxel_grids = inputs['voxel_grids']

        conv_output = self.conv_encoder(voxel_grids, batch_size=batch_size, time=time)
        out_h = self.fc(inputs, conv_output, training)

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

        # mini-batches may not be balanced, weight the losses for positive and negative examples to balance
        total_bce = class_weighted_mean_loss(label_weighted_bce, is_close_after_start)

        total_loss = total_bce

        return {
            'loss': total_loss
        }

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
            local_voxel_grid_t = make_voxelgrid_inputs_t(input_dict, local_env, local_origin_point, self.vg_info, t,
                                                         batch_size, include_robot_geometry=self.include_robot_geometry)

            local_voxel_grids_array = local_voxel_grids_array.write(t, local_voxel_grid_t)

        local_voxel_grids = tf.transpose(local_voxel_grids_array.stack(), [1, 0, 2, 3, 4, 5])
        local_voxel_grids.set_shape([None, time, None, None, None, None])  # FIXME: 2 is hardcoded here
        return local_voxel_grids

    def get_local_env(self, input_dict, batch_size):
        state_0 = {k: input_dict[add_predicted(k)][:, 0] for k in self.state_keys}

        # NOTE: to be more general, this should return a pose not just a point/position
        local_env_center = self.scenario.local_environment_center_differentiable(state_0)
        environment = {k: input_dict[k] for k in ['env', 'origin_point', 'res', 'extent']}
        local_env, local_origin_point = self.local_env_given_center(local_env_center, environment, batch_size)

        if DEBUG_AUG:
            stepper = RvizSimpleStepper()
            for b in debug_viz_batch_indices(self.batch_size):
                self.send_position_transform(local_env_center[b], 'local_env_center')
                self.send_position_transform(local_origin_point[b], 'local_origin_point')
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

    def sample_local_env_position(self, example, batch_size):
        # NOTE: for my specific implementation of state_to_local_env_pose,
        #  sampling random states and calling state_to_local_env_pose is equivalent to sampling a point in the extent
        extent = tf.reshape(example['extent'], [batch_size, 3, 2])
        extent_lower = tf.gather(extent, 0, axis=-1)
        extent_upper = tf.gather(extent, 1, axis=-1)
        local_env_center = self.aug.gen.uniform([batch_size, 3], extent_lower, extent_upper)

        return local_env_center

    def local_env_given_center(self, center_point, environment: Dict, batch_size):
        return get_local_env_and_origin_point(center_point=center_point,
                                              environment=environment,
                                              h=self.local_env_h_rows,
                                              w=self.local_env_w_cols,
                                              c=self.local_env_c_channels,
                                              indices=self.indices,
                                              batch_size=batch_size)

    def augmentation_optimization(self,
                                  inputs: Dict,
                                  batch_size,
                                  time):
        # before augmentation, get all components of the state as a set of points
        # in general this should be the swept volume, and should include the robot
        points = inputs['swept_state_and_robot_points']
        res = inputs['res']

        # sample a translation and rotation for the object state
        transformation_params = self.scenario.sample_state_augmentation_variables(10 * batch_size, self.aug.seed)
        # pick the most valid transforms, via the learned object state augmentation validity model
        predicted_errors = self.aug.invariance_model_wrapper.evaluate(transformation_params)
        best_transform_params, _ = tf.math.top_k(predicted_errors, batch_size, sorted=False)
        transformation_matrices = transformations.compose_matrix(translate=transformation_params[:3],
                                                                 angles=transformation_params[3:])

        valid, local_origin_point_aug = self.scenario.apply_state_augmentation(transformation_params,
                                                                               inputs,
                                                                               batch_size,
                                                                               time,
                                                                               self.local_env_h_rows,
                                                                               self.local_env_w_cols,
                                                                               self.local_env_c_channels)

        local_origin_point = inputs['local_origin_point']
        local_env = inputs['voxel_grids'][:, 0, :, :, :, 0]  # just use 0 because it's the same at all time steps
        local_env_occupancy = lookup_points_in_vg(points,
                                                  local_env,
                                                  res,
                                                  local_origin_point,
                                                  batch_size)

        if DEBUG_AUG:
            stepper = RvizSimpleStepper()
            for b in debug_viz_batch_indices(batch_size):
                debug_i = tf.squeeze(tf.where(1 - local_env_occupancy[b]), -1)
                points_debug_b = tf.gather(points[b], debug_i)
                self.scenario.plot_points_rviz(points_debug_b.numpy(), label='repel', color='r')

                debug_i = tf.squeeze(tf.where(local_env_occupancy[b]), -1)
                points_debug_b = tf.gather(points[b], debug_i)
                self.scenario.plot_points_rviz(points_debug_b.numpy(), label='attract', color='g')
                # stepper.step()

                send_voxelgrid_tf_origin_point_res(self.broadcaster,
                                                   origin_point=local_origin_point_aug[b],
                                                   res=res[b],
                                                   frame='local_env_aug_vg')

                bbox_msg = grid_to_bbox(rows=self.local_env_h_rows,
                                        cols=self.local_env_w_cols,
                                        channels=self.local_env_c_channels,
                                        resolution=res[b].numpy())
                bbox_msg.header.frame_id = 'local_env_aug_vg'

                self.debug.aug_bbox_pub.publish(bbox_msg)

        points_aug = transform_points_3d(transformation_matrices, points)
        valid_expanded = valid[:, None, None]
        points_aug = valid_expanded * points_aug + (1 - valid_expanded) * points

        if DEBUG_AUG:
            for b in debug_viz_batch_indices(batch_size):
                debug_i = tf.squeeze(tf.where(local_env_occupancy[b]), -1)
                points_debug_b = tf.gather(points_aug[b], debug_i)
                self.scenario.plot_points_rviz(points_debug_b.numpy(), label='attract_aug', color='g', scale=0.005)

                debug_i = tf.squeeze(tf.where(1 - local_env_occupancy[b]), -1)
                points_debug_b = tf.gather(points_aug[b], debug_i)
                self.scenario.plot_points_rviz(points_debug_b.numpy(), label='repel_aug', color='r', scale=0.005)

        new_env = self.get_new_env(inputs)
        local_env_aug = self.opt_new_env_augmentation(new_env,
                                                      points_aug,
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
                self.debug.env_aug_pub5.publish(msg)
                send_voxelgrid_tf_origin_point_res(self.broadcaster,
                                                   local_origin_point_aug[b],
                                                   res[b],
                                                   frame='local_env_aug_vg')

                self.debug_viz_state_action(inputs, b, 'aug', color='blue')
                # stepper.step()

        voxel_grids_aug = self.merge_aug_and_local_voxel_grids(local_env_aug,
                                                               inputs['voxel_grids'],
                                                               time)
        return voxel_grids_aug

    def opt_new_env_augmentation(self,
                                 new_env: Dict,
                                 points_aug,
                                 local_env_occupancy,
                                 res,
                                 local_origin_point_aug,
                                 batch_size):
        """

        Args:
            new_env: [b, h, w, c]
            points_aug: [b, n, 3], in same frame as local_origin_point_aug (i.e. robot or world frame)
                    The set of points in the swept volume of the state & robot, possibly augmented
            local_env_occupancy: [b, n]
            res: [b]
            local_origin_point_aug: [b, 3]
            batch_size: int

        Returns: [b, h, w, c]

        """
        local_env_new_center = self.sample_local_env_position(new_env, batch_size)
        local_env_new, local_env_new_origin_point = self.local_env_given_center(local_env_new_center, new_env,
                                                                                batch_size)
        # viz new env
        if DEBUG_AUG:
            for b in debug_viz_batch_indices(self.batch_size):
                self.send_position_transform(local_env_new_center[b], 'local_env_new_center')

                send_voxelgrid_tf_origin_point_res(self.broadcaster,
                                                   origin_point=local_env_new_origin_point[b],
                                                   res=res[b],
                                                   frame='local_env_new_vg')

                bbox_msg = grid_to_bbox(rows=self.local_env_h_rows,
                                        cols=self.local_env_w_cols,
                                        channels=self.local_env_c_channels,
                                        resolution=res[b].numpy())
                bbox_msg.header.frame_id = 'local_env_new_vg'

                self.debug.local_env_new_bbox_pub.publish(bbox_msg)

                env_new_dict = {
                    'env': new_env['env'][b].numpy(),
                    'res': res[b].numpy(),
                }
                msg = environment_to_vg_msg(env_new_dict, frame='new_env_aug_vg', stamp=rospy.Time(0))
                self.debug.env_aug_pub1.publish(msg)

                send_voxelgrid_tf_origin_point_res(self.broadcaster,
                                                   origin_point=new_env['origin_point'][b],
                                                   res=res[b],
                                                   frame='new_env_aug_vg')

                # Show sample new local environment, in the frame of the original local env, the one we're augmenting
                local_env_new_dict = {
                    'env': local_env_new[b].numpy(),
                    'res': res[b].numpy(),
                }
                msg = environment_to_vg_msg(local_env_new_dict, frame='local_env_aug_vg', stamp=rospy.Time(0))
                self.debug.env_aug_pub2.publish(msg)

                send_voxelgrid_tf_origin_point_res(self.broadcaster,
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
            state_points_b = points_aug[b]
            local_env_occupancy_b = local_env_occupancy[b]
            env_points_b_initial = occupied_voxels_to_points(local_env_new[b], r_b, o_b)
            env_points_b = env_points_b_initial

            initial_is_attract_indices = tf.squeeze(tf.where(local_env_occupancy_b > 0.5), 1)
            initial_attract_points_b = tf.gather(state_points_b, initial_is_attract_indices)
            if tf.size(initial_is_attract_indices) == 0:
                initial_translation_b = tf.zeros(3)
            else:
                env_points_b_initial_mean = tf.reduce_mean(env_points_b_initial, axis=0)
                initial_attract_points_b_mean = tf.reduce_mean(initial_attract_points_b, axis=0)
                initial_translation_b = initial_attract_points_b_mean - env_points_b_initial_mean
            translation_b = tf.Variable(initial_translation_b, dtype=tf.float32)
            variables = [translation_b]
            for i in range(n_steps):
                with tf.GradientTape() as tape:
                    env_points_b = env_points_b_initial + translation_b
                    is_attract_indices = tf.squeeze(tf.where(local_env_occupancy_b > 0.5), 1)
                    attract_points_b = tf.gather(state_points_b, is_attract_indices)
                    if tf.size(is_attract_indices) == 0:
                        attract_loss = 0
                        min_attract_dist_b = 0.0
                    else:
                        # NOTE: these are SQUARED distances!
                        attract_dists_b = pairwise_squared_distances(env_points_b, attract_points_b)
                        min_attract_dist_indices_b = tf.argmin(attract_dists_b, axis=1)
                        min_attract_dist_b = tf.reduce_min(attract_dists_b, axis=1)
                        nearest_attract_points = tf.gather(attract_points_b, min_attract_dist_indices_b)
                        attract_loss = tf.reduce_mean(min_attract_dist_b)

                    is_repel_indices = tf.squeeze(tf.where(local_env_occupancy_b < 0.5), 1)
                    repel_points_b = tf.gather(state_points_b, is_repel_indices)
                    if tf.size(is_repel_indices) == 0:
                        repel_loss = 0
                        min_repel_dist_b = 0.0
                    else:
                        repel_dists_b = pairwise_squared_distances(env_points_b, repel_points_b)
                        min_repel_dist_indices_b = tf.argmin(repel_dists_b, axis=1)
                        min_repel_dist_b = tf.reduce_min(repel_dists_b, axis=1)
                        nearest_repel_points = tf.gather(repel_points_b, min_repel_dist_indices_b)
                        repel_loss = tf.reduce_mean(self.barrier_func(min_repel_dist_b))

                    loss = attract_loss + repel_loss

                if DEBUG_AUG_SGD:
                    repel_close_indices = tf.squeeze(tf.where(min_repel_dist_b < self.aug.barrier_upper_lim), axis=-1)
                    nearest_repel_points_where_close = tf.gather(nearest_repel_points, repel_close_indices)
                    env_points_b_where_close = tf.gather(env_points_b, repel_close_indices)
                    if b in debug_viz_batch_indices(batch_size):
                        self.scenario.plot_points_rviz(env_points_b, label='icp', color='grey', scale=0.005)
                        self.scenario.plot_lines_rviz(nearest_attract_points, env_points_b,
                                                      label='attract_correspondence', color='g')
                        self.scenario.plot_lines_rviz(nearest_repel_points_where_close,
                                                      env_points_b_where_close,
                                                      label='repel_correspondence', color='r')
                        # stepper.step()

                gradients = tape.gradient(loss, variables)

                clipped_grads_and_vars = self.clip_env_aug_grad(gradients, variables)
                self.aug.opt.apply_gradients(grads_and_vars=clipped_grads_and_vars)

                hard_repel_constraint_satisfied = tf.reduce_min(min_repel_dist_b) > tf.square(res[b])
                hard_attract_constraint_satisfied = tf.reduce_max(min_attract_dist_b) < tf.square(res[b])
                hard_constraints_satisfied = tf.logical_and(hard_repel_constraint_satisfied,
                                                            hard_attract_constraint_satisfied)
                grad_norm = tf.linalg.norm(gradients)
                if grad_norm < self.aug.grad_norm_threshold or hard_constraints_satisfied:
                    break
            local_env_aug_b = self.points_to_voxel_grid_res_origin_point(env_points_b, r_b, o_b)

            # after local optimization, enforce the constraint by ensuring voxels with attract points are on,
            # and voxels with repel points are off
            attract_vg_b = self.points_to_voxel_grid_res_origin_point(attract_points_b, r_b, o_b)
            repel_vg_b = self.points_to_voxel_grid_res_origin_point(repel_points_b, r_b, o_b)
            # NOTE: the order of operators here is arbitrary, it gives different output, but I doubt it matters
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

    def clip_env_aug_grad(self, gradients, variables):
        def _clip(g):
            return tf.clip_by_value(g, -self.aug.grad_clip, self.aug.grad_clip)

        return [(_clip(g), v) for (g, v) in zip(gradients, variables)]

    def barrier_func(self, min_dists_b):
        return log_barrier(min_dists_b, scale=self.aug.barrier_scale, cutoff=self.aug.barrier_upper_lim)

    def debug_viz_local_env_pre_aug(self, example: Dict, time):
        local_origin_point = example['local_origin_point']
        for b in debug_viz_batch_indices(self.batch_size):
            send_voxelgrid_tf_origin_point_res(self.broadcaster,
                                               origin_point=local_origin_point[b],
                                               res=example['res'][b],
                                               frame='local_env_vg')

            bbox_msg = grid_to_bbox(rows=self.local_env_h_rows,
                                    cols=self.local_env_w_cols,
                                    channels=self.local_env_c_channels,
                                    resolution=example['res'][b].numpy())
            bbox_msg.header.frame_id = 'local_env_vg'

            self.debug.local_env_bbox_pub.publish(bbox_msg)

            self.animate_voxel_grid_states(b, example, time)

    def animate_voxel_grid_states(self, b, inputs, time):
        anim = RvizAnimationController(n_time_steps=time)
        while not anim.done:
            t = anim.t()

            local_voxel_grid_t = inputs['voxel_grids'][:, t]

            for i, state_component_k_voxel_grid in enumerate(tf.transpose(local_voxel_grid_t, [4, 0, 1, 2, 3])):
                raster_dict = {
                    'env': tf.clip_by_value(state_component_k_voxel_grid[b], 0, 1),
                    'res': inputs['res'][b].numpy(),
                }
                raster_msg = environment_to_vg_msg(raster_dict, frame='local_env_vg', stamp=rospy.Time(0))
                self.debug.raster_debug_pubs[i].publish(raster_msg)

            state_t = numpify({k: inputs[add_predicted(k)][b, t] for k in self.state_keys})
            state_t[add_predicted('joint_positions')] = inputs[add_predicted('joint_positions')][b, t]
            state_t['joint_names'] = inputs['joint_names'][b, t]
            error_msg = Float32()
            error_t = inputs['error'][b, 1]
            error_msg.data = error_t
            self.scenario.plot_state_rviz(state_t)
            self.scenario.plot_is_close(inputs['is_close'][b, 1])
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
        state_0['joint_names'] = input_dict['joint_names'][b, 0]
        action_0 = numpify({k: input_dict[k][b, 0] for k in self.action_keys})
        state_1 = numpify({k: input_dict[add_predicted(k)][b, 1] for k in self.state_keys})
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

    def send_position_transform(self, p, child: str):
        self.scenario.tf.send_transform(p, [0, 0, 0, 1], 'world', child=child, is_static=True)
