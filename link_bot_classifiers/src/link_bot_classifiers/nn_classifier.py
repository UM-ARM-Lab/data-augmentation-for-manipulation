from copy import copy
from typing import Dict

import tensorflow as tf
from colorama import Fore
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import Mean
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy, Metric

import rospy
from link_bot_classifiers.aug_opt import AugmentationOptimization
from link_bot_classifiers.classifier_debugging import ClassifierDebugging
from link_bot_classifiers.local_env_helper import LocalEnvHelper
# noinspection PyUnresolvedReferences
from link_bot_classifiers.make_voxelgrid_inputs import VoxelgridInfo
from link_bot_classifiers.robot_points import RobotVoxelgridInfo
from link_bot_data.dataset_utils import add_predicted, deserialize_scene_msg
from link_bot_data.visualization_common import make_delete_marker, make_delete_markerarray
from link_bot_pycommon.bbox_visualization import grid_to_bbox
from link_bot_pycommon.debugging_utils import debug_viz_batch_indices
from link_bot_pycommon.grid_utils import environment_to_vg_msg, \
    send_voxelgrid_tf_origin_point_res
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from link_bot_pycommon.serialization import dump_gzipped_pickle
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.classifier_losses_and_metrics import class_weighted_mean_loss
from moonshine.metrics import BinaryAccuracyOnPositives, BinaryAccuracyOnNegatives, LossMetric, \
    FalsePositiveMistakeRate, FalseNegativeMistakeRate, FalsePositiveOverallRate, FalseNegativeOverallRate
from moonshine.moonshine_utils import numpify
from moonshine.my_keras_model import MyKerasModel
from visualization_msgs.msg import MarkerArray


def debug_input():
    return rospy.get_param("DEBUG_INPUT", False)


class NNClassifier(MyKerasModel):
    def __init__(self, hparams: Dict, batch_size: int, scenario: ScenarioWithVisualization, verbose: int = 0):
        super().__init__(hparams, batch_size, verbose)
        self.scenario = scenario

        # define network structure from hparams
        self.classifier_dataset_hparams = self.hparams['classifier_dataset_hparams']
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
        self.save_inputs_path = None
        self.save_idx = 0
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

        self.local_env_helper = LocalEnvHelper(h=self.local_env_h_rows, w=self.local_env_w_cols,
                                               c=self.local_env_c_channels)
        self.debug = ClassifierDebugging(self.scenario, self.state_keys, self.action_keys)

        self.include_robot_geometry = self.hparams.get('include_robot_geometry', False)
        if not self.include_robot_geometry:
            print(Fore.RED + f"Old model! {self.include_robot_geometry=}" + Fore.RESET)
        self.robot_info = RobotVoxelgridInfo(joint_positions_key=add_predicted('joint_positions'))

        self.vg_info = VoxelgridInfo(h=self.local_env_h_rows,
                                     w=self.local_env_w_cols,
                                     c=self.local_env_c_channels,
                                     state_keys=[add_predicted(k) for k in self.points_state_keys],
                                     jacobian_follower=self.scenario.robot.jacobian_follower,
                                     robot_info=self.robot_info,
                                     include_robot_geometry=self.include_robot_geometry
                                     )

        self.aug = AugmentationOptimization(scenario=self.scenario, debug=self.debug,
                                            local_env_helper=self.local_env_helper, vg_info=self.vg_info,
                                            points_state_keys=self.points_state_keys, hparams=self.hparams,
                                            batch_size=self.batch_size, state_keys=self.state_keys,
                                            action_keys=self.action_keys)
        if self.verbose > 0:
            if self.aug.do_augmentation():
                rospy.loginfo("Using augmentation during training")
            else:
                rospy.loginfo("Not using augmentation during training")

    def preprocess_no_gradient(self, inputs, training: bool):
        batch_size = inputs['batch_size']
        time = inputs['time']

        if debug_input() and training:
            # clear the other voxel grids from previous calls
            self.debug.clear()
            self.scenario.delete_points_rviz(label='attract')
            self.scenario.delete_points_rviz(label='repel')
            self.scenario.delete_points_rviz(label='attract_aug')
            self.scenario.delete_points_rviz(label='repel_aug')
            self.scenario.delete_lines_rviz(label='attract')
            self.scenario.delete_lines_rviz(label='repel')

            for b in debug_viz_batch_indices(batch_size):
                env_b = {
                    'env':          inputs['env'][b],
                    'res':          inputs['res'][b],
                    'origin_point': inputs['origin_point'][b],
                    'extent':       inputs['extent'][b],
                }
                self.scenario.plot_environment_rviz(env_b)
                self.delete_state_action_markers('aug')
                origin_point_b = inputs['origin_point'][b].numpy().tolist()
                self.debug.send_position_transform(origin_point_b, 'origin_point')

        if training and self.aug.do_augmentation():
            # returns a copy, does NOT modify inputs in-place
            inputs = self.aug.augmentation_optimization(inputs, batch_size, time)

        local_env, local_origin_point = self.get_local_env(inputs, batch_size)

        if training and self.save_inputs_path is not None:
            self.save_inputs_path.mkdir(exist_ok=True)
            for b in range(batch_size):
                save_filename = self.save_inputs_path / f'example_{self.save_idx}.pkl.gz'
                self.save_idx += 1
                inputs_save = {
                    'rope':            inputs[add_predicted('rope')][b],
                    'joint_positions': inputs[add_predicted('joint_positions')][b],
                    'joint_names':     inputs['joint_names'][b],
                    'env':             local_env[b],
                    'res':             inputs['res'][b],
                    'origin_point':    local_origin_point[b],
                    'is_close':        inputs['is_close'][b],
                }
                with save_filename.open("wb") as file:
                    dump_gzipped_pickle(inputs_save, file)

        voxel_grids = self.vg_info.make_voxelgrid_inputs(inputs, local_env, local_origin_point, batch_size, time)
        inputs['voxel_grids'] = voxel_grids

        if debug_input() and training:
            if rospy.get_param("use_sim_time", False):
                print("use sim time is true!")
            self.debug_viz_inputs(inputs, local_origin_point, time)

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
            'precision':             Precision(),
            'recall':                Recall(),
            'accuracy on positives': BinaryAccuracyOnPositives(),
            'accuracy on negatives': BinaryAccuracyOnNegatives(),
            'loss':                  LossMetric(),
            'fp/mistakes':           FalsePositiveMistakeRate(),
            'fn/mistakes':           FalseNegativeMistakeRate(),
            'fp/total':              FalsePositiveOverallRate(),
            'fn/total':              FalseNegativeOverallRate(),
            'class_balance':         Mean(),
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
        metrics['class_balance'].update_state(labels)

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

    def get_local_env(self, inputs, batch_size):
        state_0 = {k: inputs[add_predicted(k)][:, 0] for k in self.state_keys}

        # NOTE: to be more general, this should return a pose not just a point/position
        local_env_center = self.scenario.local_environment_center_differentiable(state_0)
        local_env, local_origin_point = self.local_env_helper.get(local_env_center, inputs, batch_size)

        return local_env, local_origin_point

    def debug_viz_inputs(self, inputs: Dict, local_origin_point, time):
        for b in debug_viz_batch_indices(self.batch_size):
            if local_origin_point is not None:
                self.debug.send_position_transform(local_origin_point[b], 'local_origin_point')

                send_voxelgrid_tf_origin_point_res(self.scenario.tf.tf_broadcaster,
                                                   origin_point=local_origin_point[b],
                                                   res=inputs['res'][b],
                                                   frame='local_env_vg')

            bbox_msg = grid_to_bbox(rows=self.local_env_h_rows,
                                    cols=self.local_env_w_cols,
                                    channels=self.local_env_c_channels,
                                    resolution=numpify(inputs['res'][b]))
            bbox_msg.header.frame_id = 'local_env_vg'

            self.debug.local_env_bbox_pub.publish(bbox_msg)
            env_b = {k: inputs[k][b] for k in ['env', 'extent', 'res', 'origin_point']}
            if 'scene_msg' in inputs:
                env_b['scene_msg'] = inputs['scene_msg'][b]
            deserialize_scene_msg(env_b)
            self.scenario.plot_environment_rviz(env_b)

            self.animate_inputs(b, inputs, time)

    def animate_inputs(self, b, inputs, time):
        anim = RvizAnimationController(n_time_steps=time)
        self.debug.plot_action_rviz(inputs, b, 'inputs')
        while not anim.done:
            t = anim.t()

            if 'voxel_grids' in inputs:
                local_voxel_grid_t = inputs['voxel_grids'][:, t]

                for i, state_component_k_voxel_grid in enumerate(tf.transpose(local_voxel_grid_t, [4, 0, 1, 2, 3])):
                    raster_dict = {
                        'env': tf.clip_by_value(state_component_k_voxel_grid[b], 0, 1),
                        'res': inputs['res'][b].numpy(),
                    }
                    raster_msg = environment_to_vg_msg(raster_dict, frame='local_env_vg', stamp=rospy.Time(0))
                    self.debug.raster_debug_pubs[i].publish(raster_msg)

            self.debug.plot_state_rviz(inputs, b, t, 'inputs')

            anim.step()

    def delete_state_action_markers(self, label):
        state_delete_msg = MarkerArray(markers=[make_delete_marker(ns=label + '_l'),
                                                make_delete_marker(ns=label + 'aug_r'),
                                                make_delete_marker(ns=label + 'aug_rope')])
        self.scenario.state_viz_pub.publish(state_delete_msg)
        action_delete_msg = MarkerArray(markers=[make_delete_marker(ns=label)])
        self.scenario.action_viz_pub.publish(action_delete_msg)
        self.scenario.arrows_pub.publish(make_delete_markerarray())
