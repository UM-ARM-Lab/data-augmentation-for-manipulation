#!/usr/bin/env python
import pathlib
from typing import Dict, List, Optional, Tuple

import tensorflow as tf
from colorama import Fore
from tensorflow import keras
from tensorflow.keras import layers

import rospy
from jsk_recognition_msgs.msg import BoundingBox
from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_data.dataset_utils import add_predicted
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.grid_utils import batch_idx_to_point_3d_in_env_tf, \
    batch_point_to_idx_tf_3d_in_batched_envs
from link_bot_pycommon.pycommon import make_dict_float32, make_dict_tf_float32
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from moonshine.classifier_losses_and_metrics import binary_classification_sequence_metrics_function, \
    class_weighted_mean_loss
from moonshine.get_local_environment import get_local_env_and_origin_3d_tf as get_local_env
from moonshine.moonshine_utils import add_batch, remove_batch, sequence_of_dicts_to_dict_of_tensors
from moonshine.my_keras_model import MyKerasModel
from moonshine.raster_3d import raster_3d
from mps_shape_completion_msgs.msg import OccupancyStamped

DEBUG_VIZ = False


class NNClassifier(MyKerasModel):
    def __init__(self, hparams: Dict, batch_size: int, scenario: ScenarioWithVisualization):
        super().__init__(hparams, batch_size)
        self.scenario = scenario

        self.raster_debug_pubs = [
            rospy.Publisher(f'classifier_raster_debug_{i}', OccupancyStamped, queue_size=10, latch=False) for i in
            range(4)]
        self.local_env_bbox_pub = rospy.Publisher('local_env_bbox', BoundingBox, queue_size=10, latch=True)

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
            self.uncertainty_head = layers.Dense(self.certs_k, activation=None)
            # self.uncertainty_head = keras.Sequential([layers.Dense(128, activation='relu'),
            #                                           layers.Dense(128, activation='relu'),
            #                                           layers.Dense(self.certs_k, activation=None),
            #                                           ])

    def make_traj_voxel_grids_from_input_dict(self, input_dict: Dict, batch_size, time: int):
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

        conv_outputs_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        debug_info_seq = []
        for t in tf.range(time):
            state_t = {k: input_dict[add_predicted(k)][:, t] for k in self.state_keys}

            local_env_center_t = self.scenario.local_environment_center_differentiable(state_t)
            # by converting too and from the frame of the full environment, we ensure the grids are aligned
            indices = batch_point_to_idx_tf_3d_in_batched_envs(local_env_center_t, input_dict)
            local_env_center_t = batch_idx_to_point_3d_in_env_tf(*indices, input_dict)

            local_env_t, local_env_origin_t = get_local_env(center_point=local_env_center_t,
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
            for i, (k, state_component_t) in enumerate(state_t.items()):
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

            out_conv_z = self.fwd_conv(batch_size, local_voxel_grid_t)

            conv_outputs_array = conv_outputs_array.write(t, out_conv_z)

            if DEBUG_VIZ:
                debug_info_seq.append((state_t, local_env_origin_t, local_env_t, local_voxel_grid_t))

        conv_outputs = conv_outputs_array.stack()
        return tf.transpose(conv_outputs, [1, 0, 2]), debug_info_seq

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

        # mask out / ignore examples where is_close [0] is 0
        is_close_at_start = dataset_element['is_close'][:, 0]
        bce = bce * is_close_at_start

        # weight examples by the perception reliability, loss reliable means less important to predict, so lower loss
        if 'perception_reliability' in dataset_element and self.hparams.get('use_perception_reliability_loss', False):
            perception_reliability_weight = dataset_element['perception_reliability']
        else:
            perception_reliability_weight = tf.ones_like(bce)
        perception_reliability_weighted_bce = bce * perception_reliability_weight

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
        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true=tf.zeros_like(outputs['uncertainty']), y_pred=outputs['uncertainty'], from_logits=True))
        diversity = tf.reduce_mean(tf.square(tf.matmul(tf.transpose(w), w) - tf.eye(self.certs_k)))
        return loss + diversity

    def compute_metrics(self, dataset_element, outputs):
        m = binary_classification_sequence_metrics_function(dataset_element, outputs)
        m['uncertainty'] = self.orthogonal_certificates_uncertainty_loss(outputs)
        return m

    @tf.function
    def call(self, input_dict: Dict, training, **kwargs):
        batch_size = input_dict['batch_size']
        time = tf.cast(input_dict['time'], tf.int32)

        conv_output, debug_info_seq = self.make_traj_voxel_grids_from_input_dict(input_dict, batch_size, time)

        out_h = self.fc(conv_output, input_dict, training)

        if self.hparams.get('uncertainty_head', False):
            out_uncertainty = self.uncertainty_head(out_h)

        # for every timestep's output, map down to a single scalar, the logit for accept probability
        all_accept_logits = self.output_layer(out_h)
        # ignore the first output, it is meaningless to predict the validity of a single state
        valid_accept_logits = all_accept_logits[:, 1:]
        valid_accept_probabilities = self.sigmoid(valid_accept_logits)

        if DEBUG_VIZ:
            self.debug_rviz(input_dict, debug_info_seq)

        return {
            'logits':        valid_accept_logits,
            'probabilities': valid_accept_probabilities,
            'uncertainty':   out_uncertainty,
        }

    def fc(self, conv_output, input_dict, training):
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

    def debug_rviz(self, input_dict: Dict, debug_info_seq: List[Tuple]):
        import pickle
        from merrrt_visualization.rviz_animation_controller import RvizSimpleStepper
        from link_bot_pycommon.bbox_visualization import grid_to_bbox
        from moonshine.moonshine_utils import numpify
        from moonshine.indexing import index_dict_of_batched_tensors_tf, index_time_with_metadata
        from link_bot_pycommon.grid_utils import environment_to_occupancy_msg, send_occupancy_tf
        import numpy as np
        from time import time

        debug_filename = f'debug_{int(time())}.pkl'
        print(f"Saving debug info to {debug_filename}")
        with open(debug_filename, "wb") as debug_file:
            pickle.dump({'input_dict': input_dict, 'debug_info': debug_info_seq}, debug_file)

        stepper = RvizSimpleStepper()
        batch_size = input_dict.pop("batch_size").numpy().astype(np.int32)
        input_dict.pop("time")

        for b in range(batch_size):
            example = index_dict_of_batched_tensors_tf(input_dict, b)

            for t, debug_info_t in enumerate(debug_info_seq):
                state_t, local_env_origin_t, local_env_t, local_voxel_grid_t = debug_info_t
                for i, state_component_k_voxel_grid in enumerate(tf.transpose(local_voxel_grid_t, [4, 0, 1, 2, 3])):
                    raster_dict = {
                        'env':    tf.clip_by_value(state_component_k_voxel_grid[b], 0, 1),
                        'origin': local_env_origin_t[b].numpy(),
                        'res':    input_dict['res'][b].numpy(),
                    }
                    raster_msg = environment_to_occupancy_msg(raster_dict, frame='local_occupancy')
                    self.raster_debug_pubs[i].publish(raster_msg)

                local_env_dict = {
                    'env':    local_env_t[b],
                    'origin': local_env_origin_t[b].numpy(),
                    'res':    input_dict['res'][b].numpy(),
                }
                send_occupancy_tf(self.scenario.tf.tf_broadcaster, local_env_dict, frame='local_occupancy')

                pred_t = index_time_with_metadata({}, example, self.pred_state_keys, t)
                action_t = {k: example[k][t] for k in self.action_keys}
                self.scenario.plot_state_rviz(numpify(pred_t), label='predicted', color='#0000ffff')
                if action_t is not None:
                    self.scenario.plot_action_rviz(numpify(pred_t), numpify(action_t), label='action',
                                                   color='#0000ffff')
                # # Ground-Truth
                # true_t = index_time_with_metadata({}, example, self.true_state_keys, t)
                # self.scenario.plot_state_rviz(numpify(true_t), label='actual', color='#ff0000ff', scale=1.1)
                # label_t = example['is_close'][1]
                # self.scenario.plot_is_close(label_t)
                bbox_msg = grid_to_bbox(rows=self.local_env_h_rows,
                                        cols=self.local_env_w_cols,
                                        channels=self.local_env_c_channels,
                                        resolution=example['res'].numpy())
                bbox_msg.header.frame_id = 'local_occupancy'
                self.local_env_bbox_pub.publish(bbox_msg)

                stepper.step()


# FIXME: inherit from Ensemble
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

        self.nets = []
        # FIXME: hack
        for model_dir in [path]:
            net = net_class_name(hparams=self.hparams, batch_size=batch_size, scenario=scenario)

            ckpt = tf.train.Checkpoint(model=net)
            manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=1)

            status = ckpt.restore(manager.latest_checkpoint).expect_partial()
            if manager.latest_checkpoint:
                print(Fore.CYAN + "Restored from {}".format(manager.latest_checkpoint) + Fore.RESET)
                if manager.latest_checkpoint:
                    status.assert_existing_objects_matched()
            else:
                raise RuntimeError(f"Failed to restore {manager.latest_checkpoint}!!!")

            self.nets.append(net)

            self.state_keys = net.state_keys
            self.action_keys = net.action_keys
            self.true_state_keys = net.true_state_keys
            self.pred_state_keys = net.pred_state_keys

    def check_constraint_from_example(self, example: Dict, training: Optional[bool] = False):
        predictions = [net(net.preprocess_no_gradient(example, training), training=training) for net in self.nets]
        predictions_dict = sequence_of_dicts_to_dict_of_tensors(predictions)
        mean_predictions = {k: tf.math.reduce_mean(v, axis=0) for k, v in predictions_dict.items()}
        stdev_predictions = {k: tf.math.reduce_std(v, axis=0) for k, v in predictions_dict.items()}
        return mean_predictions, stdev_predictions

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
        net_inputs.update(make_dict_tf_float32(environment))

        for action_key in self.action_keys:
            net_inputs[action_key] = tf.cast(actions[action_key], tf.float32)

        for state_key in self.state_keys:
            planned_state_key = add_predicted(state_key)
            net_inputs[planned_state_key] = tf.cast(states[state_key], tf.float32)

        if self.hparams['stdev']:
            net_inputs[add_predicted('stdev')] = tf.cast(states['stdev'], tf.float32)

        net_inputs = make_dict_tf_float32(net_inputs)
        mean_predictions, stdev_predictions = self.check_constraint_from_example(net_inputs, training=False)
        mean_probability = mean_predictions['probabilities']
        stdev_probability = stdev_predictions['probabilities']
        mean_probability = tf.squeeze(mean_probability, axis=2)
        stdev_probability = tf.squeeze(stdev_probability, axis=2)
        return mean_probability, stdev_probability

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
        mean_probabilities, stdev_probabilities = self.check_constraint_tf_batched(environment=environment,
                                                                                   states=states_sequence_dict,
                                                                                   actions=actions_dict,
                                                                                   batch_size=1,
                                                                                   state_sequence_length=state_sequence_length)
        mean_probabilities = remove_batch(mean_probabilities)
        stdev_probabilities = remove_batch(stdev_probabilities)
        return mean_probabilities, stdev_probabilities

    def check_constraint(self,
                         environment: Dict,
                         states_sequence: List[Dict],
                         actions: List[Dict]):
        states_sequence = [make_dict_float32(s) for s in states_sequence]
        mean_probabilities, stdev_probabilities = self.check_constraint_tf(environment=environment,
                                                                           states_sequence=states_sequence,
                                                                           actions=actions)
        mean_probabilities = mean_probabilities.numpy()
        stdev_probabilities = stdev_probabilities.numpy()
        return mean_probabilities, stdev_probabilities

    @staticmethod
    def get_net_class():
        return NNClassifier
