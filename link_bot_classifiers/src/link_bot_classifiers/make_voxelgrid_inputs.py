from dataclasses import dataclass
from typing import List

import tensorflow as tf

from link_bot_data.dataset_utils import add_predicted
from moonshine.raster_3d import batch_points_to_voxel_grid_res_origin_point


@dataclass
class MakeVoxelgridInfo:
    batch_size: int
    h: int
    w: int
    c: int
    state_keys: List[str]


def make_voxelgrid_inputs_t(
        inputs,
        local_env,
        local_origin_point,
        info: MakeVoxelgridInfo,
        t,
        include_robot_voxels=False):
    state_t = {k: inputs[k][:, t] for k in info.state_keys}
    local_voxel_grid_t_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    # Insert the environment as channel 0
    local_voxel_grid_t_array = local_voxel_grid_t_array.write(0, local_env)
    # insert the rastered states
    for channel_idx, (k, state_component_t) in enumerate(state_t.items()):
        n_points_in_component = int(state_component_t.shape[1] / 3)
        points = tf.reshape(state_component_t, [info.batch_size, -1, 3])
        flat_batch_indices = tf.repeat(tf.range(info.batch_size, dtype=tf.int64), n_points_in_component, axis=0)
        flat_points = tf.reshape(points, [-1, 3])
        flat_points.set_shape([n_points_in_component * info.batch_size, 3])
        flat_res = tf.repeat(inputs['res'], n_points_in_component, axis=0)
        flat_origin_point = tf.repeat(local_origin_point, n_points_in_component, axis=0)
        state_component_voxel_grid = batch_points_to_voxel_grid_res_origin_point(flat_batch_indices,
                                                                                 flat_points,
                                                                                 flat_res,
                                                                                 flat_origin_point,
                                                                                 info.h,
                                                                                 info.w,
                                                                                 info.c,
                                                                                 info.batch_size)
        local_voxel_grid_t_array = local_voxel_grid_t_array.write(channel_idx + 1, state_component_voxel_grid)
        # insert the rastered robot state
        # could have the points saved to disc, load them up and transform them based on the current robot state?
        # (maybe by resolution? we could have multiple different resolutions)
    if include_robot_voxels:
        raise NotImplementedError()
        # robot_voxel_grid = make_robot_voxel_grid(inputs, t, batch_size)
        # local_voxel_grid_t_array = local_voxel_grid_t_array.write(channel_idx + 1, robot_voxel_grid)
        # n_channels = len(self.state_keys) + 2
    else:
        n_channels = len(info.state_keys) + 1
    local_voxel_grid_t = tf.transpose(local_voxel_grid_t_array.stack(), [1, 2, 3, 4, 0])
    # add channel dimension information because tf.function erases it?
    local_voxel_grid_t.set_shape([None, None, None, None, n_channels])
    return local_voxel_grid_t