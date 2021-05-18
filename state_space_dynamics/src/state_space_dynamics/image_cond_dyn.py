from typing import Dict

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow import keras

from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.grid_utils import batch_idx_to_point_3d_in_env_tf, batch_point_to_idx_tf_3d_in_batched_envs
from moonshine.get_local_environment import get_local_env_and_origin_3d_tf_old as get_local_env
from moonshine.matrix_operations import batch_outer_product
from moonshine.moonshine_utils import sequence_of_dicts_to_dict_of_tensors
from moonshine.raster_3d import raster_3d
from moonshine.my_keras_model import MyKerasModel
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


class ImageCondDynamics(MyKerasModel):

    def __init__(self, hparams: Dict, batch_size: int, scenario: ExperimentScenario):
        super().__init__(hparams, batch_size)
        self.scenario = scenario

        # # BEGIN DEBUG
        # from jsk_recognition_msgs.msg import BoundingBox
        # self.debug_pub = rospy.Publisher('classifier_debug', OccupancyStamped, queue_size=10, latch=True)
        # self.local_env_bbox_pub = rospy.Publisher('local_env_bbox', BoundingBox, queue_size=10, latch=True)
        # # END DEBUG

        self.rope_image_k = self.hparams['rope_image_k']
        self.local_env_h_rows = self.hparams['local_env_h_rows']
        self.local_env_w_cols = self.hparams['local_env_w_cols']
        self.local_env_c_channels = self.hparams['local_env_c_channels']

        self.concat = layers.Concatenate()
        self.concat2 = layers.Concatenate()

        # State keys is all the things we want the model to take in/predict
        self.state_keys = self.hparams['state_keys']
        self.action_keys = self.hparams['action_keys']
        self.dataset_states_description = self.hparams['dynamics_dataset_hparams']['states_description']
        self.dataset_actions_description = self.hparams['dynamics_dataset_hparams']['action_description']
        self.state_dimensions = [self.dataset_states_description[k] for k in self.state_keys]
        self.total_state_dimensions = sum(self.state_dimensions)

        self.state_action_dense_layers = []
        if 'state_action_only_fc_layer_sizes' in self.hparams:
            for fc_layer_size in self.hparams['state_action_only_fc_layer_sizes']:
                self.state_action_dense_layers.append(layers.Dense(fc_layer_size, activation='relu'))

        self.conv_only_dense_layers = []
        if 'conv_only_fc_layer_sizes' in self.hparams:
            if len(self.hparams['conv_only_fc_layer_sizes']) > 0:
                for fc_layer_size in self.hparams['conv_only_fc_layer_sizes'][:-1]:
                    self.conv_only_dense_layers.append(layers.Dense(fc_layer_size, activation='relu'))
                self.conv_only_dense_layers.append(layers.Dense(
                    self.hparams['conv_only_fc_layer_sizes'][-1], activation=None))

        self.final_dense_layers = []
        final_fc_layer_sizes = []
        if 'final_fc_layer_sizes' in self.hparams:
            final_fc_layer_sizes = self.hparams['final_fc_layer_sizes']
        elif 'fc_layer_sizes' in self.hparams:
            final_fc_layer_sizes = self.hparams['fc_layer_sizes']
        for fc_layer_size in final_fc_layer_sizes:
            self.final_dense_layers.append(layers.Dense(fc_layer_size, activation='relu'))
        self.final_dense_layers.append(layers.Dense(self.total_state_dimensions, activation=None))

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

        # How to combine the image hidden vector and state/action hidden vector
        def outer_product_then_flatten(args):
            state_action, conv_z_t = args
            outer_product = batch_outer_product(state_action, conv_z_t)
            return tf.reshape(outer_product, [self.batch_size, -1])

        self.combine_state_action_and_conv_output = layers.Lambda(outer_product_then_flatten)

        self.flatten_conv_output = layers.Flatten()

    def compute_loss(self, dataset_element, outputs):
        return {
            'loss': self.scenario.dynamics_loss_function(dataset_element, outputs)
        }

    def compute_metrics(self, dataset_element, outputs):
        return self.scenario.dynamics_metrics_function(dataset_element, outputs)

    # @tf.function
    def call(self, example, training, mask=None):
        batch_size = example['batch_size']
        time = example[self.action_keys[0]].shape[1] + 1

        s_0 = {k: example[k][:, 0] for k in self.state_keys}

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
        #     'env': example['env'][b],
        #     'origin': example['origin'][b],
        #     'res': example['res'][b],
        #     'extent': example['extent'][b],
        # }
        # self.scenario.plot_environment_rviz(full_env_dict)
        # # END DEBUG

        pred_states = [s_0]
        for t in range(time - 1):
            s_t = pred_states[-1]
            s_t_local = self.scenario.put_state_local_frame(s_t)
            action_t = {k: example[k][:, t] for k in self.action_keys}
            local_action_t = self.scenario.put_action_local_frame(s_t, action_t)

            local_env_center_t = self.scenario.local_environment_center_differentiable(s_t)

            # by converting too and from the frame of the full environment, we ensure the grids are aligned
            indices = batch_point_to_idx_tf_3d_in_batched_envs(local_env_center_t, example)
            local_env_center_t = batch_idx_to_point_3d_in_env_tf(*indices, example)

            local_env_t, local_env_origin_t = get_local_env(center_point=local_env_center_t,
                                                            full_env=example['env'],
                                                            full_env_origin=example['origin'],
                                                            res=example['res'],
                                                            local_h_rows=self.local_env_h_rows,
                                                            local_w_cols=self.local_env_w_cols,
                                                            local_c_channels=self.local_env_c_channels,
                                                            batch_x_indices=batch_x_indices,
                                                            batch_y_indices=batch_y_indices,
                                                            batch_z_indices=batch_z_indices,
                                                            batch_size=batch_size)

            local_voxel_grid_t_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
            local_voxel_grid_t_array = local_voxel_grid_t_array.write(0, local_env_t)
            for i, state_component_t in enumerate(s_t.values()):
                state_component_voxel_grid = raster_3d(state=state_component_t,
                                                       pixel_indices=pixel_indices,
                                                       res=example['res'],
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
            # local_env_dict = {
            #     'env': tf.clip_by_value(tf.reduce_sum(local_voxel_grid_t[b], axis=-1), 0, 1),
            #     # 'env': local_voxel_grid_t[b],
            #     'origin': local_env_origin_t[b].numpy(),
            #     'res': example['res'][b].numpy(),
            # }
            # msg = environment_to_occupancy_msg(local_env_dict, frame='local_occupancy')
            # link_bot_sdf_utils.send_occupancy_tf(self.scenario.broadcaster, local_env_dict, frame='local_occupancy')
            # self.debug_pub.publish(msg)
            # self.scenario.plot_state_rviz(numpify(index_dict_of_batched_vectors_tf(s_t, b)), label='pred')
            # local_extent = compute_extent_3d(*local_voxel_grid_t[b].shape[:3], resolution=example['res'][b].numpy())
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

            # CNN
            z_t = local_voxel_grid_t
            for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
                z_t = conv_layer(z_t)
                z_t = pool_layer(z_t)
            conv_z_t = self.flatten_conv_output(z_t)
            for dense_layer in self.conv_only_dense_layers:
                conv_z_t = dense_layer(conv_z_t)

            # concat into one big state-action vector
            states_and_actions = list(s_t_local.values()) + list(local_action_t.values())
            state_action_t = self.concat2(states_and_actions)
            for dense_layer in self.state_action_dense_layers:
                state_action_t = dense_layer(state_action_t)

            # combine state/action vector with output of CNN
            full_z_t = self.combine_state_action_and_conv_output([state_action_t, conv_z_t])

            # dense layers for combined vector
            for dense_layer in self.final_dense_layers:
                full_z_t = dense_layer(full_z_t)

            delta_s_t = self.vector_to_state_dict(full_z_t)
            s_t_plus_1 = self.scenario.integrate_dynamics(s_t, delta_s_t)

            pred_states.append(s_t_plus_1)

        pred_states_dict = sequence_of_dicts_to_dict_of_tensors(pred_states, axis=1)
        return pred_states_dict

    def vector_to_state_dict(self, z):
        start_idx = 0
        state_vectors = []
        for dim in self.state_dimensions:
            state_vectors.append(z[:, start_idx:start_idx + dim])
            start_idx += dim
        return dict(zip(self.state_keys, state_vectors))


class ImageCondDynamicsWrapper(BaseDynamicsFunction):

    def make_net_and_checkpoint(self, batch_size, scenario):
        net = ImageCondDynamics(hparams=self.hparams, batch_size=batch_size, scenario=scenario)
        ckpt = tf.train.Checkpoint(model=net)
        return net, ckpt


model = ImageCondDynamics
