from copy import copy
from typing import Dict

import tensorflow as tf
from colorama import Fore
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.metrics import Metric

import rospy
from jsk_recognition_msgs.msg import BoundingBox
from link_bot_classifiers.augmentation_optimization import AugmentationOptimization
from link_bot_classifiers.classifier_debugging import ClassifierDebugging
from link_bot_classifiers.local_env_helper import LocalEnvHelper
from link_bot_classifiers.make_voxelgrid_inputs import VoxelgridInfo
from link_bot_classifiers.robot_points import RobotVoxelgridInfo
from link_bot_pycommon.debugging_utils import debug_viz_batch_indices
from link_bot_pycommon.pycommon import densify_points
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from merrrt_visualization.rviz_animation_controller import RvizSimpleStepper
from moonshine.get_local_environment import create_env_indices, get_local_env_and_origin_point
from moonshine.metrics import LossMetric
from moonshine.my_keras_model import MyKerasModel
from rviz_voxelgrid_visuals_msgs.msg import VoxelgridStamped

DEBUG_INPUT = False
DEBUG_AUG = False
DEBUG_AUG_SGD = False


class NNRecoveryModel(MyKerasModel):
    def __init__(self, hparams: Dict, batch_size: int, scenario: ScenarioWithVisualization):
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
        self.include_robot_geometry = self.hparams.get('include_robot_geometry', False)

        self.state_keys = self.hparams['state_keys']
        self.points_state_keys = copy(self.state_keys)
        self.points_state_keys.remove("joint_positions")  # FIXME: feels hacky
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

        self.local_env_helper = LocalEnvHelper(h=self.local_env_h_rows, w=self.local_env_w_cols,
                                               c=self.local_env_c_channels)
        self.debug = ClassifierDebugging(self.scenario, self.state_keys, self.action_keys)

        self.indices = self.create_env_indices(batch_size)

        self.include_robot_geometry = self.hparams.get('include_robot_geometry', False)
        print(Fore.LIGHTBLUE_EX + f"{self.include_robot_geometry=}" + Fore.RESET)

        self.robot_info = RobotVoxelgridInfo(joint_positions_key='joint_positions')

        self.vg_info = VoxelgridInfo(h=self.local_env_h_rows,
                                     w=self.local_env_w_cols,
                                     c=self.local_env_c_channels,
                                     state_keys=[k for k in self.points_state_keys],
                                     jacobian_follower=self.scenario.robot.jacobian_follower,
                                     robot_info=self.robot_info,
                                     include_robot_geometry=self.include_robot_geometry,
                                     )
        self.aug = AugmentationOptimization(scenario=self.scenario, debug=self.debug,
                                            local_env_helper=self.local_env_helper, vg_info=self.vg_info,
                                            points_state_keys=self.points_state_keys, hparams=self.hparams,
                                            batch_size=self.batch_size, action_keys=self.action_keys,
                                            state_keys=self.state_keys)
        if self.aug.do_augmentation():
            rospy.loginfo("Using augmentation during training")
        else:
            rospy.loginfo("Not using augmentation during training")

    def preprocess_no_gradient(self, inputs, training: bool):
        batch_size = inputs['batch_size']

        # this should already be in the input, I'm worried it's giving the wrong values sometimes though
        # inputs['origin_point'] = batch_extent_to_origin_point_tf(inputs['extent'], inputs['res'])

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
        local_env, local_origin_point = self.get_local_env(inputs)

        local_voxel_grid_t = self.vg_info.make_voxelgrid_inputs_t(inputs, local_env, local_origin_point, 0, batch_size)

        inputs['voxel_grids'] = local_voxel_grid_t
        inputs['local_origin_point'] = local_origin_point

        inputs['swept_state_and_robot_points'] = self.compute_swept_state_and_robot_points(inputs)

        if DEBUG_AUG:
            self.debug_viz_local_env_pre_aug(inputs)

        if training and self.aug.do_augmentation():
            self.aug.augmentation_optimization(inputs, batch_size, time=2)

        return inputs

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

    def call(self, input_dict: Dict, training, **kwargs):
        state = {k: input_dict[k][:, 0] for k in self.state_keys}
        state_in_local_frame = self.scenario.put_state_local_frame(state)
        state_lf_list = list(state_in_local_frame.values())
        action = {k: input_dict[k][:, 0] for k in self.action_keys}
        action = self.scenario.put_action_local_frame(state, action)
        action_list = list(action.values())
        state_in_robot_frame = self.scenario.put_state_robot_frame(state)
        state_rf_list = list(state_in_robot_frame.values())

        batch_size = input_dict['batch_size']
        voxel_grids = input_dict['voxel_grids']
        conv_output = self.conv_encoder(voxel_grids, batch_size=batch_size)

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

    def compute_swept_state_and_robot_points(self, inputs):
        batch_size = inputs['batch_size']

        def _make_points(k, t):
            v = inputs[k][:, t]
            points = tf.reshape(v, [batch_size, -1, 3])
            points = densify_points(batch_size, points)
            return points

        state_points = {k: _make_points(k, 0) for k in self.points_state_keys}

        return state_points

    def conv_encoder(self, voxel_grids, batch_size):
        conv_z = voxel_grids
        for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
            conv_h = conv_layer(conv_z)
            conv_z = pool_layer(conv_h)
        out_conv_z = conv_z
        out_conv_z_dim = out_conv_z.shape[1] * out_conv_z.shape[2] * out_conv_z.shape[3] * out_conv_z.shape[4]
        out_conv_z = tf.reshape(out_conv_z, [batch_size, out_conv_z_dim])
        return out_conv_z

    def create_env_indices(self, batch_size: int):
        return create_env_indices(self.local_env_h_rows, self.local_env_w_cols, self.local_env_c_channels, batch_size)

    def get_local_env(self, input_dict):
        state_0 = {k: input_dict[k][:, 0] for k in self.state_keys}

        # NOTE: to be more general, this should return a pose not just a point/position
        local_env_center = self.scenario.local_environment_center_differentiable(state_0)
        environment = {k: input_dict[k] for k in ['env', 'origin_point', 'res', 'extent']}
        local_env, local_origin_point = self.local_env_given_center(local_env_center, environment)

        if DEBUG_AUG:
            stepper = RvizSimpleStepper()
            for b in debug_viz_batch_indices(self.batch_size):
                self.send_position_transform(local_env_center[b], 'local_env_center')
                self.send_position_transform(local_origin_point[b], 'local_origin_point')
                # stepper.step()

        return local_env, local_origin_point

    def local_env_given_center(self, center_point, environment: Dict):
        return get_local_env_and_origin_point(center_point=center_point,
                                              environment=environment,
                                              h=self.local_env_h_rows,
                                              w=self.local_env_w_cols,
                                              c=self.local_env_c_channels,
                                              indices=self.indices,
                                              batch_size=self.batch_size)
