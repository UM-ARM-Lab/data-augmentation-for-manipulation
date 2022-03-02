from typing import Dict

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.metrics import Metric

import rospy
from jsk_recognition_msgs.msg import BoundingBox
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.grid_utils import batch_extent_to_origin_point_tf
from link_bot_pycommon.pycommon import dgather
from moonshine.get_local_environment_tf import get_local_env_and_origin_point, create_env_indices
from moonshine.metrics import LossMetric
from moonshine.my_keras_model import MyKerasModel
from moonshine.raster_3d import points_to_voxel_grid_res_origin_point_batched
from rviz_voxelgrid_visuals_msgs.msg import VoxelgridStamped

DEBUG_VIZ = False

class NNRecoveryModel(MyKerasModel):
    def __init__(self, hparams: Dict, batch_size: int, scenario: ExperimentScenario):
        super().__init__(hparams, batch_size)
        self.scenario = scenario

        self.debug_pub = rospy.Publisher('classifier_debug', VoxelgridStamped, queue_size=10, latch=True)
        self.raster_debug_pub = rospy.Publisher('classifier_raster_debug', VoxelgridStamped, queue_size=10, latch=True)
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

    def preprocess_no_gradient(self, example, training: bool):
        example['origin_point'] = batch_extent_to_origin_point_tf(example['extent'], example['res'])
        return example

    def compute_loss(self, dataset_element, outputs):
        y_true = dataset_element['recovery_probability'][:, 1:2]  # 1:2 instead of just 1 to preserve the shape
        y_pred = outputs['logits']
        loss = tf.keras.losses.binary_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=True)
        # target recovery_probability examples are weighted higher because there are so few of them
        # when y_true is 1 this term goes to infinity (high weighting), when y_true is 0 it equals 1 (normal weighting)
        l = tf.math.divide_no_nan(-1.0, y_true - 1)
        loss = loss * l
        return {
            'loss': tf.reduce_mean(loss)
        }

    def create_metrics(self):
        super().create_metrics()
        return {
            'loss': LossMetric(),
        }

    def compute_metrics(self, metrics: Dict[str, Metric], losses: Dict, dataset_element, outputs):
        pass

    @tf.function
    def call(self, input_dict: Dict, training, **kwargs):
        batch_size = input_dict['batch_size']

        conv_output = self.make_voxelgrid_inputs(input_dict, batch_size)

        state = {k: input_dict[k][:, 0] for k in self.state_keys}
        state_in_local_frame = self.scenario.put_state_local_frame(state)
        state_lf_list = list(state_in_local_frame.values())
        action = {k: input_dict[k][:, 0] for k in self.action_keys}
        action = self.scenario.put_action_local_frame(state, action)
        action_list = list(action.values())
        state_in_robot_frame = self.scenario.put_state_robot_frame(state)
        state_rf_list = list(state_in_robot_frame.values())

        if 'with_robot_frame' not in self.hparams:
            print("no hparam 'with_robot_frame'. This must be an old model!")
            concat_args = [conv_output] + state_lf_list + action_list
        elif self.hparams['with_robot_frame']:
            concat_args = [conv_output] + state_rf_list + state_lf_list + action_list
        else:
            concat_args = [conv_output] + state_lf_list + action_list

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
            'logits':        logits,
            'probabilities': probabilities,
        }

    def make_voxelgrid_inputs(self, input_dict: Dict, batch_size, time: int = 1):
        # Construct a [b, h, w, c, 3] grid of the indices which make up the local environment
        indices = self.create_env_indices(batch_size)

        if DEBUG_VIZ:
            # plot the occupancy grid
            time_steps = np.arange(time)
            b = 0
            full_env_dict = {
                'env':    input_dict['env'][b],
                'origin': input_dict['origin'][b],
                'res':    input_dict['res'][b],
                'extent': input_dict['extent'][b],
            }
            self.scenario.plot_environment_rviz(full_env_dict)

        state = {k: input_dict[k][:, 0] for k in self.state_keys}

        local_env_center = self.scenario.local_environment_center_differentiable(state)

        env = dgather(input_dict, ['env', 'origin_point', 'res'])
        local_env, local_origin_point = get_local_env_and_origin_point(center_point=local_env_center,
                                                                       environment=env,
                                                                       h=self.local_env_h_rows,
                                                                       w=self.local_env_w_cols,
                                                                       c=self.local_env_c_channels,
                                                                       indices=indices,
                                                                       batch_size=batch_size)

        local_voxel_grid_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        local_voxel_grid_array = local_voxel_grid_array.write(0, local_env)
        for i, state_component in enumerate(state.values()):
            n_points_in_component = int(state_component.shape[1] / 3)
            points = tf.reshape(state_component, [batch_size, -1, 3])
            flat_batch_indices = tf.repeat(tf.range(batch_size, dtype=tf.int64), n_points_in_component, axis=0)
            flat_points = tf.reshape(points, [-1, 3])
            flat_points.set_shape([n_points_in_component * self.batch_size, 3])
            flat_res = tf.repeat(input_dict['res'], n_points_in_component, axis=0)
            flat_origin_point = tf.repeat(local_origin_point, n_points_in_component, axis=0)
            state_component_voxel_grid = points_to_voxel_grid_res_origin_point_batched(flat_batch_indices,
                                                                                       flat_points,
                                                                                       flat_res,
                                                                                       flat_origin_point,
                                                                                       self.local_env_h_rows,
                                                                                       self.local_env_w_cols,
                                                                                       self.local_env_c_channels,
                                                                                       batch_size)

            local_voxel_grid_array = local_voxel_grid_array.write(i + 1, state_component_voxel_grid)
        local_voxel_grid = tf.transpose(local_voxel_grid_array.stack(), [1, 2, 3, 4, 0])
        # add channel dimension information because tf.function erases it somehow...
        local_voxel_grid.set_shape([None, None, None, None, len(self.state_keys) + 1])

        conv_z = local_voxel_grid
        for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
            conv_h = conv_layer(conv_z)
            conv_z = pool_layer(conv_h)
        out_conv_z = conv_z
        out_conv_z_dim = out_conv_z.shape[1] * out_conv_z.shape[2] * out_conv_z.shape[3] * out_conv_z.shape[4]
        out_conv_z = tf.reshape(out_conv_z, [batch_size, out_conv_z_dim])
        return out_conv_z

    def create_env_indices(self, batch_size: int):
        return create_env_indices(self.local_env_h_rows, self.local_env_w_cols, self.local_env_c_channels, batch_size)

