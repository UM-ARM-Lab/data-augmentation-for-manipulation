from typing import Dict

import numpy as np
import tensorflow as tf
from deprecated import deprecated

import rospy
from geometry_msgs.msg import TransformStamped
from link_bot_pycommon.grid_utils_np import compute_extent_3d, extent_res_to_origin_point
from moonshine.numpify import numpify
from moonshine.tensorflow_utils import swap_xy


def pad_voxel_grid(voxel_grid, origin, res, new_shape):
    assert voxel_grid.shape[0] <= new_shape[0]
    assert voxel_grid.shape[1] <= new_shape[1]
    assert voxel_grid.shape[2] <= new_shape[2]

    h_pad1 = tf.math.floor((new_shape[0] - voxel_grid.shape[0]) / 2)
    h_pad2 = tf.math.ceil((new_shape[0] - voxel_grid.shape[0]) / 2)

    w_pad1 = tf.math.floor((new_shape[1] - voxel_grid.shape[1]) / 2)
    w_pad2 = tf.math.ceil((new_shape[1] - voxel_grid.shape[1]) / 2)

    c_pad1 = tf.math.floor((new_shape[2] - voxel_grid.shape[2]) / 2)
    c_pad2 = tf.math.ceil((new_shape[2] - voxel_grid.shape[2]) / 2)

    padded_env = tf.pad(voxel_grid, paddings=[[h_pad1, h_pad2], [w_pad1, w_pad2], [c_pad1, c_pad2]])
    new_origin = origin + [h_pad1, w_pad1, c_pad1]
    new_extent = compute_extent_3d(new_shape[0], new_shape[1], new_shape[2], res, new_origin)

    return padded_env, new_origin, new_extent


def batch_extent_to_env_size_tf(extent_3d):
    extent_3d = tf.reshape(extent_3d, [-1, 3, 2])
    min = tf.gather(extent_3d, 0, axis=-1)
    max = tf.gather(extent_3d, 1, axis=-1)
    return tf.abs(max - min)


def dist_to_bbox(point, lower, upper):
    return tf.maximum(tf.maximum(tf.reduce_max(point - upper), 0), tf.maximum(tf.reduce_max(lower - point), 0))


def binary_or(a, b):
    return tf.clip_by_value(a + b, 0, 1)


def binary_and(a, b):
    return a * b


def subtract(a, b):
    return tf.clip_by_value(a - b, 0, 1)


def lookup_points_in_vg(state_points, env, res, origin_point, batch_size):
    """
    Returns the values of env at state_points
    Args:
        state_points: [b, n, 3], in same frame as origin_point
        env: [b, h, w, c]
        res:
        origin_point: [b, 3] in same frame as state_points
        batch_size:

    Returns: [b, n]

    """
    n_points = state_points.shape[1]
    vg_indices = batch_point_to_idx(state_points,
                                    tf.expand_dims(res, axis=1),
                                    tf.expand_dims(origin_point, axis=1))
    batch_indices = tf.tile(tf.range(batch_size, dtype=tf.int64)[:, None, None], [1, n_points, 1])
    batch_and_vg_indices = tf.concat([batch_indices, vg_indices], axis=-1)
    occupancy_at_state_points = tf.gather_nd(env, batch_and_vg_indices)
    return occupancy_at_state_points


def occupied_voxels_to_points(vg, res, origin_point):
    indices = tf.where(vg > 0.5)
    occupied_points = batch_idx_to_point_3d_tf_res_origin_point(indices, res, origin_point)
    return occupied_points


def occupied_voxels_to_points_batched(vg, res, origin_point):
    all_indices = tf.where(vg > 0.5)
    batch_indices = all_indices[:, 0]
    indices = all_indices[:, 1:]
    res_gathered = tf.gather(res, batch_indices, axis=0)
    origin_point_gathered = tf.gather(origin_point, batch_indices, axis=0)
    occupied_points = batch_idx_to_point_3d_tf_res_origin_point(indices, res_gathered, origin_point_gathered)
    return batch_indices, occupied_points


def round_to_res(x, res):
    # helps with stupid numerics issues
    return tf.cast(tf.round(x / res), tf.int64)


def batch_align_to_grid_tf(point, origin_point, res):
    """

    Args:
        point: [n, 3], meters, in the same frame as origin_point
        origin_point: [n, 3], meters, in the same frame as point
        res: [n], meters

    Returns:

    """
    res_expanded = tf.expand_dims(res, axis=-1)
    return tf.cast(tf.round((point - origin_point) / res_expanded), tf.float32) * res_expanded + origin_point


def voxel_grid_distance(a, b):
    return tf.abs(a - b)


def indeces_to_point(rowcols, resolution, origin):
    return (rowcols - origin) * resolution


def batch_idx_to_point_3d_tf_res_origin_point(indices, res, origin_point):
    indices_xyz = swap_xy(indices)
    return tf.cast(indices_xyz, tf.float32) * tf.expand_dims(res, axis=-1) + origin_point


def batch_idx_to_point_3d_in_env_tf_res_origin(row, col, channel, res, origin):
    origin = tf.cast(origin, tf.int64)
    y = tf.cast(row - tf.gather(origin, 0, axis=-1), tf.float32) * res
    x = tf.cast(col - tf.gather(origin, 1, axis=-1), tf.float32) * res
    z = tf.cast(channel - tf.gather(origin, 2, axis=-1), tf.float32) * res
    return tf.stack([x, y, z], axis=-1)


def batch_idx_to_point_3d_in_env_tf(row, col, channel, env: Dict):
    return batch_idx_to_point_3d_in_env_tf_res_origin(row, col, channel, env['res'], env['origin'])


def idx_to_point_3d_tf(row: int,
                       col: int,
                       channel: int,
                       resolution: float,
                       origin: np.ndarray):
    y = (row - origin[0]) * resolution
    x = (col - origin[1]) * resolution
    z = (channel - origin[2]) * resolution
    return tf.stack([x, y, z], axis=-1)


def batch_extent_to_env_shape_xyz_tf(extent, res):
    extent = tf.cast(extent, tf.float32)
    res = tf.cast(res, tf.float32)
    env_size = batch_extent_to_env_size_tf(extent)
    return round_to_res(env_size, tf.expand_dims(res, axis=-1))


def batch_extent_to_center_tf(extent_3d):
    extent_3d = tf.reshape(extent_3d, [-1, 3, 2])
    return tf.reduce_mean(extent_3d, axis=-1)


def batch_center_res_shape_to_origin_point(center, res, h, w, c):
    shape_xyz = tf.stack([w, h, c], axis=-1)
    return center - (tf.cast(shape_xyz, tf.float32) * tf.expand_dims(res, axis=-1) / 2)


@deprecated
def batch_extent_to_origin_point_tf(extent, res):
    center_xyz = batch_extent_to_center_tf(extent_3d=extent)
    shape_xyz = batch_extent_to_env_shape_xyz_tf(extent=extent, res=res)
    return center_xyz - (tf.cast(shape_xyz, tf.float32) * tf.expand_dims(res, axis=-1) / 2)


def send_voxelgrid_tf(broadcaster, environment: Dict, frame: str = 'vg'):
    send_voxelgrid_tf_extent_res(broadcaster, environment['extent'], environment['res'], frame)


def send_voxelgrid_tf_extent_res(broadcaster, extent, res, frame: str = 'vg'):
    origin_point = extent_res_to_origin_point(extent, res)
    send_voxelgrid_tf_origin_point_res(broadcaster, origin_point, res, frame)


def send_voxelgrid_tf_origin_point_res(broadcaster,
                                       origin_point,
                                       res,
                                       child_frame_id: str = 'vg',
                                       parent_frame_id: str = 'world'):
    origin_xyz = origin_point - (res / 2)
    origin_xyz = numpify(origin_xyz)
    # the rviz plugin displays the boxes with the corner at the given translation, not the center
    # but the origin_point is at the center, so this offsets things correctly

    transform = TransformStamped()
    transform.header.stamp = rospy.Time.now()
    transform.header.frame_id = parent_frame_id
    transform.child_frame_id = child_frame_id
    transform.transform.translation.x = origin_xyz[0]
    transform.transform.translation.y = origin_xyz[1]
    transform.transform.translation.z = origin_xyz[2]
    transform.transform.rotation.x = 0
    transform.transform.rotation.y = 0
    transform.transform.rotation.z = 0
    transform.transform.rotation.w = 1
    broadcaster.sendTransform(transform)


def batch_point_to_idx_tf(x,
                          y,
                          resolution: float,
                          origin):
    col = round_to_res(x, resolution + origin[1])
    row = round_to_res(y, resolution + origin[0])
    return row, col


def batch_point_to_idx(points, res, origin_point):
    """

    Args:
        points: [b,3] points in a frame, call it world
        res: [b] meters
        origin_point: [b,3] the position [x,y,z] of the center of the voxel (0,0,0) in the same frame as points

    Returns:

    """
    return swap_xy(round_to_res((points - origin_point), tf.expand_dims(res, axis=-1)))


def batch_point_to_idx_tf_3d_res_origin(points, res, origin):
    x = tf.gather(points, 0, axis=-1)
    y = tf.gather(points, 1, axis=-1)
    z = tf.gather(points, 2, axis=-1)
    col = tf.cast(x / res + tf.gather(origin, 1, axis=-1), tf.int64)
    row = tf.cast(y / res + tf.gather(origin, 0, axis=-1), tf.int64)
    channel = tf.cast(z / res + tf.gather(origin, 2, axis=-1), tf.int64)
    return row, col, channel


def batch_point_to_idx_tf_3d_in_batched_envs(points, env: Dict):
    return batch_point_to_idx(points, env['res'], env['origin_point'])


def batch_point_to_idx_tf_3d(x,
                             y,
                             z,
                             resolution: float,
                             origin):
    col = round_to_res(x, resolution + origin[1])
    row = round_to_res(y, resolution + origin[0])
    channel = round_to_res(z, resolution + origin[2])
    return row, col, channel
