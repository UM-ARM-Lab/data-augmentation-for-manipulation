from dataclasses import dataclass
from typing import List

import pyjacobian_follower
import tensorflow as tf

from link_bot_classifiers.robot_points import batch_transform_robot_points, RobotVoxelgridInfo
from link_bot_data.dataset_utils import add_predicted
from link_bot_pycommon.base_dual_arm_rope_scenario import densify_points
from moonshine.moonshine_utils import numpify
from moonshine.raster_3d import points_to_voxel_grid_res_origin_point_batched


@dataclass
class VoxelgridInfo:
    h: int
    w: int
    c: int
    state_keys: List[str]
    jacobian_follower: pyjacobian_follower.JacobianFollower
    robot_info: RobotVoxelgridInfo


def make_robot_voxelgrid(inputs, local_origin_point, t, info: VoxelgridInfo, batch_size):
    robot_points = make_robot_points_batched(batch_size, info, inputs, t)
    n_robot_points = robot_points.shape[1]
    flat_batch_indices = tf.repeat(tf.range(batch_size, dtype=tf.int64), n_robot_points, axis=0)
    flat_points = tf.reshape(robot_points, [-1, 3])
    flat_points.set_shape([n_robot_points * batch_size, 3])
    flat_res = tf.repeat(inputs['res'], n_robot_points, axis=0)
    flat_origin_point = tf.repeat(local_origin_point, n_robot_points, axis=0)
    robot_voxelgrid = points_to_voxel_grid_res_origin_point_batched(flat_batch_indices,
                                                                    flat_points,
                                                                    flat_res,
                                                                    flat_origin_point,
                                                                    info.h,
                                                                    info.w,
                                                                    info.c,
                                                                    batch_size)
    return robot_voxelgrid


def make_robot_points_batched(batch_size, info: VoxelgridInfo, inputs, t):
    names = inputs['joint_names'][:, t]
    positions = inputs[info.robot_info.joint_positions_key][:, t]
    robot_points = batch_transform_robot_points(info.jacobian_follower,
                                                numpify(names),
                                                positions.numpy(),
                                                info.robot_info,
                                                batch_size)
    return robot_points


def make_voxelgrid_inputs_t(
        inputs,
        local_env,
        local_origin_point,
        info: VoxelgridInfo,
        t,
        batch_size,
        include_robot_geometry=False):
    # TODO: move all the work of converting state into matrices of points to preprocess_no_gradient
    state_t = {k: inputs[k][:, t] for k in info.state_keys}
    local_voxel_grid_t_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    # Insert the environment as channel 0
    local_voxel_grid_t_array = local_voxel_grid_t_array.write(0, local_env)
    # insert the rastered states
    channel_idx = 1
    for (k, state_component_t) in state_t.items():
        points = tf.reshape(state_component_t, [batch_size, -1, 3])
        num_densify = 5
        points = densify_points(batch_size=batch_size, points=points, num_densify=num_densify)
        n_points_in_component = max(int(state_component_t.shape[1] / 3 * num_densify) - num_densify, 1)
        flat_points = tf.reshape(points, [-1, 3])
        flat_batch_indices = tf.repeat(tf.range(batch_size, dtype=tf.int64), n_points_in_component, axis=0)
        flat_points.set_shape([n_points_in_component * batch_size, 3])
        flat_res = tf.repeat(inputs['res'], n_points_in_component, axis=0)
        flat_origin_point = tf.repeat(local_origin_point, n_points_in_component, axis=0)
        state_component_voxel_grid = points_to_voxel_grid_res_origin_point_batched(flat_batch_indices,
                                                                                   flat_points,
                                                                                   flat_res,
                                                                                   flat_origin_point,
                                                                                   info.h,
                                                                                   info.w,
                                                                                   info.c,
                                                                                   batch_size)
        local_voxel_grid_t_array = local_voxel_grid_t_array.write(channel_idx, state_component_voxel_grid)
        channel_idx += 1

    if include_robot_geometry:
        robot_voxel_grid = make_robot_voxelgrid(inputs, local_origin_point, t, info, batch_size)
        local_voxel_grid_t_array = local_voxel_grid_t_array.write(channel_idx, robot_voxel_grid)
        n_channels = len(info.state_keys) + 2
    else:
        n_channels = len(info.state_keys) + 1

    local_voxel_grid_t = tf.transpose(local_voxel_grid_t_array.stack(), [1, 2, 3, 4, 0])

    # add channel dimension information because tf.function erases it?
    local_voxel_grid_t.set_shape([None, None, None, None, n_channels])
    # NOTE: for debugging the cases where the robot intersects the environment, which shouldn't really happen
    # robot_env_intersections = tf.reduce_sum(
    #     tf.reduce_sum(tf.reduce_sum(local_voxel_grid_t[..., 0] * local_voxel_grid_t[..., 4], -1), -1), -1)
    return local_voxel_grid_t
