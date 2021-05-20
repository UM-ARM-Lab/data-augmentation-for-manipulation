import tensorflow as tf

from link_bot_pycommon.grid_utils import batch_point_to_idx_tf_3d_res_origin, batch_point_to_idx


@tf.function
def raster_3d_wrapped(state, pixel_indices, res, origin, h, w, c, k, batch_size):
    return raster_3d(state, pixel_indices, res, origin, h, w, c, k, batch_size)


def raster_3d(state, pixel_indices, res, origin, h, w, c, k, batch_size):
    """
    Args:
        state: [batch_size, 3*k]
        pixel_indices: [batch_size, h, w, c, 3]
        res: [batch_size]
        origin: [batch_size, 3]
        h:
        w:
        c:
        k:
        batch_size:

    Returns: 1-channel voxel grid, using the max to reduce values contributed to by different points in the input state
    """
    res_expanded = res[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
    n_points = tf.cast(state.shape[1] / 3, tf.int64)
    batch_size = tf.cast(batch_size, tf.int64)
    points = tf.reshape(state, [batch_size, n_points, 3])

    # shape [batch_size, h, w, c, 3]
    origin_expanded = origin[:, tf.newaxis, tf.newaxis, tf.newaxis]
    pixel_centers_y_x_z = (pixel_indices - origin_expanded) * res_expanded

    # swap x and y, so it goes y=row, x=col, z=channel which is our indexing convention
    points_y_x_z = tf.stack([points[:, :, 1], points[:, :, 0], points[:, :, 2]], axis=-1)  # [b, n, 3]
    tiled_points_y_x_z = points_y_x_z[:, tf.newaxis, tf.newaxis, tf.newaxis]  # [b, 1, 1, 1, n, 3]
    tiled_points_y_x_z = tf.tile(tiled_points_y_x_z, [1, h, w, c, 1, 1])  # [b, h, w, c, n, 3]
    squared_distances = tf.reduce_sum(tf.square(tf.expand_dims(pixel_centers_y_x_z, axis=4) - tiled_points_y_x_z),
                                      axis=-1)  # [b, h, w, c, n]
    local_voxel_grid_for_points = tf.exp(-k * squared_distances)
    local_voxel_grid = tf.math.reduce_max(local_voxel_grid_for_points, axis=-1)
    return local_voxel_grid


# this function is slower with tf.function, don't use it
@tf.function
def points_to_voxel_grid_wrapped(batch_indices, points, res, origin, h, w, c, batch_size):
    return points_to_voxel_grid(batch_indices, points, res, origin, h, w, c, batch_size)


def points_to_voxel_grid(batch_indices, points, res, origin, h, w, c, batch_size):
    """
    Args:
        batch_indices: [n], batch_indices[i] is the batch indices for point points[i]. Must be int64 type
        points: [n, 3]
        res: [n]
        origin: [n, 3]
        h:
        w:
        c:
        batch_size:

    Returns: 1-channel binary voxel grid
    """
    n = points.shape[0]
    indices = batch_point_to_idx_tf_3d_res_origin(points, res, origin)  # [n, 4]
    indices = tf.stack([batch_indices, *indices], axis=-1)
    ones = tf.ones([n])
    voxel_grid = tf.scatter_nd(indices, ones, [batch_size, h, w, c])
    voxel_grid = tf.clip_by_value(voxel_grid, 0, 1)
    return voxel_grid


def batch_points_to_voxel_grid_res_origin_point(batch_indices, points, res, origin_point, h, w, c, batch_size):
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

    Returns: 1-channel binary voxel grid
    """
    n = points.shape[0]
    indices = batch_point_to_idx(points, res, origin_point)  # [n, 4]
    rows, cols, channels = tf.unstack(indices, axis=-1)
    indices = tf.stack([batch_indices, rows, cols, channels], axis=-1)
    ones = tf.ones([n])
    voxel_grid = tf.scatter_nd(indices, ones, [batch_size, h, w, c])
    voxel_grid = tf.clip_by_value(voxel_grid, 0, 1)
    return voxel_grid


def points_to_voxel_grid_res_origin_point(points, res, origin_point, h, w, c):
    """
    Args:
        points: [n, 3]
        res: [n]
        origin_point: [n, 3]
        h:
        w:
        c:

    Returns: 1-channel binary voxel grid
    """
    n = points.shape[0]
    indices = batch_point_to_idx(points, res, origin_point)  # [n, 3]
    ones = tf.ones([n])
    voxel_grid = tf.scatter_nd(indices, ones, [h, w, c])
    voxel_grid = tf.clip_by_value(voxel_grid, 0, 1)
    return voxel_grid
