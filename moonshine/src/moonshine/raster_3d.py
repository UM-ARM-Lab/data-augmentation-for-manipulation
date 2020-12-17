import tensorflow as tf


@tf.function
def raster_3d(state, pixel_indices, res, origin, h, w, c, k, batch_size: int):
    """
    output is 1-channel voxel grid, using the max to reduce values contributed to by different points in the input state
    """
    res = res[0]
    n_points = tf.cast(state.shape[1] / 3, tf.int64)
    batch_size = tf.cast(batch_size, tf.int64)
    points = tf.reshape(state, [batch_size, n_points, 3])

    # shape [batch_size, h, w, c, 3]
    origin_expanded = origin[:, tf.newaxis, tf.newaxis, tf.newaxis]
    pixel_centers_y_x_z = (pixel_indices - origin_expanded) * res

    # swap x and y, so it goes y=row, x=col, z=channel which is our indexing convention
    points_y_x_z = tf.stack([points[:, :, 1], points[:, :, 0], points[:, :, 2]], axis=-1)  # [b, n, 3]
    tiled_points_y_x_z = points_y_x_z[:, tf.newaxis, tf.newaxis, tf.newaxis]  # [b, 1, 1, 1, n, 3]
    tiled_points_y_x_z = tf.tile(tiled_points_y_x_z, [1, h, w, c, 1, 1])  # [b, h, w, c, n, 3]
    squared_distances = tf.reduce_sum(tf.square(tf.expand_dims(pixel_centers_y_x_z, axis=4) - tiled_points_y_x_z),
                                      axis=-1)  # [b, h, w, c, n]
    local_voxel_grid_for_points = tf.exp(-k * squared_distances)
    local_voxel_grid = tf.math.reduce_max(local_voxel_grid_for_points, axis=-1)
    return local_voxel_grid
