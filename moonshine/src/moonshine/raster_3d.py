import numpy as np
import tensorflow as tf

from link_bot_pycommon.grid_utils import idx_to_point_3d


@tf.function
def raster_3d(state, pixel_indices, res, origin, h, w, c, k, batch_size: int):
    """ output is 1-channel voxel grid, using the max to reduce values contributed to by different points in the input state """
    res = res[0]
    n_points = tf.cast(state.shape[1] / 3, tf.int64)
    batch_size = tf.cast(batch_size, tf.int64)
    points = tf.reshape(state, [batch_size, n_points, 3])

    # Below is a un-vectorized implementation, which is much easier to read and understand
    # rope_images = np.zeros([batch_size, h, w, c], dtype=np.float32)
    # for batch_index in range(batch_size):
    #     for point_idx in range(n_points):
    #         for row, col, channel in np.ndindex(h, w, c):
    #             point_in_meters = points[batch_index, point_idx]
    #             pixel_center_in_meters = idx_to_point_3d(row, col, channel, res, origin[batch_index])
    #             squared_distance = np.sum(np.square(point_in_meters - pixel_center_in_meters))
    #             pixel_value = np.exp(-k * squared_distance)
    #             pixel_value = max(rope_images[batch_index, row, col, channel], pixel_value)
    #             rope_images[batch_index, row, col, channel] = pixel_value
    # rope_images = np.expand_dims(rope_images, axis=4)

    # this is how we construct an empty grid of shape [b, h, w, c]

    # shape [batch_size, h, w, c, 3]
    origin_expanded = origin[:, tf.newaxis, tf.newaxis, tf.newaxis]
    pixel_centers_y_x_z = (pixel_indices - origin_expanded) * res

    local_voxel_grid = tf.identity(pixel_centers_y_x_z[:, :, :, :, 0]) * 0
    for point_idx in tf.range(n_points):
        point_y_x_z = tf.stack([points[:, point_idx, 1], points[:, point_idx, 0], points[:, point_idx, 2]], axis=1)
        tiled_point_y_x_z = point_y_x_z[:, tf.newaxis, tf.newaxis, tf.newaxis]
        tiled_point_y_x_z = tf.tile(tiled_point_y_x_z, [1, h, w, c, 1])
        # swap x and y

        squared_distances = tf.reduce_sum(tf.square(pixel_centers_y_x_z - tiled_point_y_x_z), axis=4)
        local_voxel_grid_for_point = tf.exp(-k * squared_distances)
        local_voxel_grid = tf.maximum(local_voxel_grid,  local_voxel_grid_for_point)
    return local_voxel_grid
