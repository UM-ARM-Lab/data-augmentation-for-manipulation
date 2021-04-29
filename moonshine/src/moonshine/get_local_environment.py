from dataclasses import dataclass

import tensorflow as tf


@dataclass
class EnvIndices:
    batch_x: tf.Tensor
    batch_y: tf.Tensor
    batch_z: tf.Tensor
    pixels: tf.Tensor


def create_env_indices(local_env_h_rows: int, local_env_w_cols: int, local_env_c_channels: int, batch_size: int):
    # Construct a [b, h, w, c, 3] grid of the indices which make up the local environment
    pixel_row_indices = tf.range(0, local_env_h_rows, dtype=tf.float32)
    pixel_col_indices = tf.range(0, local_env_w_cols, dtype=tf.float32)
    pixel_channel_indices = tf.range(0, local_env_c_channels, dtype=tf.float32)
    x_indices, y_indices, z_indices = tf.meshgrid(pixel_col_indices, pixel_row_indices, pixel_channel_indices)
    # Make batched versions for creating the local environment
    batch_y_indices = tf.cast(tf.tile(tf.expand_dims(y_indices, axis=0), [batch_size, 1, 1, 1]), tf.int64)
    batch_x_indices = tf.cast(tf.tile(tf.expand_dims(x_indices, axis=0), [batch_size, 1, 1, 1]), tf.int64)
    batch_z_indices = tf.cast(tf.tile(tf.expand_dims(z_indices, axis=0), [batch_size, 1, 1, 1]), tf.int64)

    # Convert for rastering state
    pixel_indices = tf.stack([y_indices, x_indices, z_indices], axis=3)
    pixel_indices = tf.expand_dims(pixel_indices, axis=0)
    pixel_indices = tf.tile(pixel_indices, [batch_size, 1, 1, 1, 1])

    return EnvIndices(batch_x=batch_x_indices,
                      batch_y=batch_y_indices,
                      batch_z=batch_z_indices,
                      pixels=pixel_indices)


# FIXME: use origin_point, not origin, origin is not as precise.
def get_local_env_and_origin_3d_tf(center_point,
                                   full_env,
                                   full_env_origin,
                                   res,
                                   local_h_rows: int,
                                   local_w_cols: int,
                                   local_c_channels: int,
                                   batch_x_indices,
                                   batch_y_indices,
                                   batch_z_indices,
                                   batch_size: int):
    """
    :param center_point: [batch, 3]
    :param full_env: [batch, h, w, c]
    :param full_env_origin: [batch, 3]
    :param res: [batch]
    :param local_h_rows: scalar
    :param local_w_cols: scalar
    :return:
    """
    full_env_origin = tf.convert_to_tensor(full_env_origin, dtype=tf.float32)

    local_center = tf.stack([local_h_rows / 2, local_w_cols / 2, local_c_channels / 2], axis=0)

    center_cols = center_point[:, 0] / res + full_env_origin[:, 1]
    center_rows = center_point[:, 1] / res + full_env_origin[:, 0]
    center_channels = center_point[:, 2] / res + full_env_origin[:, 2]

    center_point_coordinates = tf.stack([center_rows, center_cols, center_channels], axis=1)
    local_env_origin = full_env_origin - center_point_coordinates + local_center
    local_to_full_offset = tf.cast(full_env_origin - local_env_origin, tf.int64)

    # Transform into coordinate of the full_env
    batch_y_indices_in_full_env_frame = batch_y_indices + local_to_full_offset[:, 0, tf.newaxis, tf.newaxis, tf.newaxis]
    batch_x_indices_in_full_env_frame = batch_x_indices + local_to_full_offset[:, 1, tf.newaxis, tf.newaxis, tf.newaxis]
    batch_z_indices_in_full_env_frame = batch_z_indices + local_to_full_offset[:, 2, tf.newaxis, tf.newaxis, tf.newaxis]

    tile_sizes = [1, local_h_rows, local_w_cols, local_c_channels]
    batch_int64 = tf.cast(batch_size, tf.int64)
    batch_indices = tf.tile(tf.range(0, batch_int64, dtype=tf.int64)[:, tf.newaxis, tf.newaxis, tf.newaxis], tile_sizes)
    gather_indices = tf.stack(
        [batch_indices, batch_y_indices_in_full_env_frame, batch_x_indices_in_full_env_frame,
         batch_z_indices_in_full_env_frame],
        axis=4)
    local_env = tf.gather_nd(full_env, gather_indices)

    return local_env, local_env_origin


def get_local_env_and_origin_2d_tf(center_point,
                                   full_env,
                                   full_env_origin,
                                   res,
                                   local_h_rows: int,
                                   local_w_cols: int):
    """
    :param center_point: [batch, 2]
    :param full_env: [batch, h, w]
    :param full_env_origin: [batch, 2]
    :param res: [batch]
    :param local_h_rows: scalar
    :param local_w_cols: scalar
    :return:
    """
    batch_size = full_env.shape[0]
    full_h_rows = full_env.shape[1]
    full_w_cols = full_env.shape[2]

    local_center = tf.stack([local_h_rows / 2, local_w_cols / 2], axis=0)
    full_center = tf.stack([full_h_rows / 2, full_w_cols / 2], axis=0)

    center_cols = center_point[:, 0] / res + full_env_origin[:, 1]
    center_rows = center_point[:, 1] / res + full_env_origin[:, 0]
    center_point_coordinates = tf.stack([center_rows, center_cols], axis=1)
    local_env_origin = full_env_origin - center_point_coordinates + local_center
    local_to_full_offset = tf.cast(full_center - local_env_origin, tf.int64)

    # Add batch
    # noinspection PyUnresolvedReferences
    batch_y_indices = tf.tile(tf.expand_dims(y_indices, axis=0), [batch_size, 1, 1])
    # noinspection PyUnresolvedReferences
    batch_x_indices = tf.tile(tf.expand_dims(x_indices, axis=0), [batch_size, 1, 1])
    # Transform into coordinate of the full_env
    batch_y_indices_in_full_env_frame = batch_y_indices + local_to_full_offset[:, 0, tf.newaxis, tf.newaxis]
    batch_x_indices_in_full_env_frame = batch_x_indices + local_to_full_offset[:, 1, tf.newaxis, tf.newaxis]

    batch_indices = tf.tile(tf.range(0, batch_size, dtype=tf.int64)[
                            :, tf.newaxis, tf.newaxis], [1, local_h_rows, local_w_cols])
    gather_indices = tf.stack([batch_indices, batch_y_indices_in_full_env_frame,
                               batch_x_indices_in_full_env_frame], axis=3)
    image = tf.expand_dims(full_env, axis=3)
    local_image = tf.gather_nd(image, gather_indices)
    local_env = tf.transpose(local_image[:, :, :, 0], [0, 2, 1])

    return local_env, local_env_origin
