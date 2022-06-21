from math import pi
from typing import Dict

import numpy as np
import tensorflow as tf
from tensorflow_graphics.geometry.transformation.rotation_matrix_3d import from_euler


def swap_xy(x):
    """

    Args:
        x: has shape [b1, b2, ..., bn, 3]
        n_batch_dims: same as n in the above shape, number of dimensions before the dimension of 3 (x,y,z)

    Returns: the x/y will be swapped

    """
    first = tf.gather(x, 0, axis=-1)
    second = tf.gather(x, 1, axis=-1)
    z = tf.gather(x, 2, axis=-1)
    swapped = tf.stack([second, first, z], axis=-1)
    return swapped


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


def round_to_res(x, res):
    # helps with stupid numerics issues
    return tf.cast(tf.round(x / res), tf.int64)


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


def homogeneous(points):
    return tf.concat([points, tf.ones_like(points[..., 0:1])], axis=-1)


def transform_points_3d(transform_matrix, points):
    """

    Args:
        transform_matrix: [b1, b2, ..., 4, 4]
        points: [b1, b2, ..., 3]

    Returns:

    """
    points_homo = homogeneous(points)
    points_homo = tf.expand_dims(points_homo, axis=-1)
    transformed_points = tf.matmul(transform_matrix, points_homo)
    return tf.squeeze(transformed_points, axis=-1)[..., :3]


def xyzrpy_to_matrices(params):
    """

    Args:
        params:  [b1,b2,...,6] in the form [x,y,z,roll,pitch,yaw]

    Returns: [b1,b2,...,4,4] with the assumption of roll, pitch, yaw, then translation (aka the normal thing)

    """
    translation = params[..., :3][..., None]
    angles = params[..., 3:]
    r33 = from_euler(angles)
    r34 = tf.concat([r33, translation], axis=-1)
    bottom_row = tf.constant([0, 0, 0, 1], dtype=tf.float32)
    bottom_row = tf.ones(params.shape[:-1] + [1, 4], tf.float32) * bottom_row
    matrices = tf.concat([r34, bottom_row], axis=-2)
    return matrices


# GENERATED BY SYMPY
def transformation_jacobian(params):
    """

    Args:
        params:  [b1,b2,...,6]

    Returns:

    """
    x, y, z, roll, pitch, yaw = tf.unstack(params, axis=-1)
    ones = tf.ones_like(x)
    zeros = tf.zeros_like(x)
    jacobian = tf.stack([
        [
            [zeros, zeros, zeros, ones],
            [zeros, zeros, zeros, zeros],
            [zeros, zeros, zeros, zeros],
            [zeros, zeros, zeros, zeros]
        ],
        [
            [zeros, zeros, zeros, zeros],
            [zeros, zeros, zeros, ones],
            [zeros, zeros, zeros, zeros],
            [zeros, zeros, zeros, zeros]
        ],
        [
            [zeros, zeros, zeros, zeros],
            [zeros, zeros, zeros, zeros],
            [zeros, zeros, zeros, ones],
            [zeros, zeros, zeros, zeros]
        ],
        [
            [zeros, tf.sin(pitch) * tf.cos(roll) * tf.cos(yaw) + tf.sin(roll) * tf.sin(yaw),
             -tf.sin(pitch) * tf.sin(roll) * tf.cos(yaw) + tf.sin(yaw) * tf.cos(roll), zeros],
            [zeros, tf.sin(pitch) * tf.sin(yaw) * tf.cos(roll) - tf.sin(roll) * tf.cos(yaw),
             -tf.sin(pitch) * tf.sin(roll) * tf.sin(yaw) - tf.cos(roll) * tf.cos(yaw), zeros],
            [zeros, tf.cos(pitch) * tf.cos(roll), -tf.sin(roll) * tf.cos(pitch), zeros],
            [zeros, zeros, zeros, zeros]
        ],
        [
            [-tf.sin(pitch) * tf.cos(yaw), tf.sin(roll) * tf.cos(pitch) * tf.cos(yaw),
             tf.cos(pitch) * tf.cos(roll) * tf.cos(yaw), zeros],
            [-tf.sin(pitch) * tf.sin(yaw), tf.sin(roll) * tf.sin(yaw) * tf.cos(pitch),
             tf.sin(yaw) * tf.cos(pitch) * tf.cos(roll), zeros],
            [-tf.cos(pitch), -tf.sin(pitch) * tf.sin(roll), -tf.sin(pitch) * tf.cos(roll), zeros],
            [zeros, zeros, zeros, zeros]
        ],
        [
            [-tf.sin(yaw) * tf.cos(pitch), -tf.sin(pitch) * tf.sin(roll) * tf.sin(yaw) - tf.cos(roll) * tf.cos(yaw),
             -tf.sin(pitch) * tf.sin(yaw) * tf.cos(roll) + tf.sin(roll) * tf.cos(yaw), zeros],
            [tf.cos(pitch) * tf.cos(yaw), tf.sin(pitch) * tf.sin(roll) * tf.cos(yaw) - tf.sin(yaw) * tf.cos(roll),
             tf.sin(pitch) * tf.cos(roll) * tf.cos(yaw) + tf.sin(roll) * tf.sin(yaw), zeros],
            [zeros, zeros, zeros, zeros],
            [zeros, zeros, zeros, zeros]
        ]
    ])  # [6,4,4,b1,b2,...]
    batch_axes = list(np.arange(len(params.shape) - 1) + 3)
    jacobian = tf.transpose(jacobian, batch_axes + [0, 1, 2])
    return jacobian


def euler_angle_diff(euler1, euler2):
    abs_diff = tf.abs(euler1 - euler2)
    return tf.minimum(abs_diff, 2 * pi - abs_diff)


def points_to_voxel_grid_res_origin_point_batched(batch_indices, points, res, origin_point, h, w, c, batch_size):
    """
    Args:
        batch_indices: [n], batch_indices[i] is the batch indices for point points[i]. Must be int64 type
        points: [n, 3]
        res: [n]
        origin_point: [n, 3]
        h:
        w:
        c:
        batch_size:

    Returns: 1-channel binary voxel grid of shape [b,h,w,c]
    """
    n = points.shape[0]
    indices = batch_point_to_idx(points, res, origin_point)  # [n, 4]
    rows, cols, channels = tf.unstack(indices, axis=-1)
    indices = tf.stack([batch_indices, rows, cols, channels], axis=-1)
    ones = tf.ones([n])
    voxel_grid = tf.scatter_nd(indices, ones, [batch_size, h, w, c])
    voxel_grid = tf.clip_by_value(voxel_grid, 0, 1)
    return voxel_grid


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
