import tensorflow as tf


def make_rotation_matrix_like(x, theta):
    # https://en.wikipedia.org/wiki/Rotation_matrix
    z = tf.zeros_like(theta)
    one = tf.ones_like(theta)
    rotation_matrix = tf.stack([tf.stack([tf.cos(theta), -tf.sin(theta), z], axis=-1),
                                tf.stack([tf.sin(theta), tf.cos(theta), z], axis=-1),
                                tf.stack([z, z, one], axis=-1)], axis=-2)
    return rotation_matrix


def rotate_points_3d(rotation_matrix, points):
    """

    Args:
        rotation_matrix: [b1, b2, ..., b2, 3, 3]
        points: [b1, b2, ..., b2, 3]

    Returns:

    """
    rotated_points = tf.matmul(rotation_matrix, tf.expand_dims(points, axis=-1))
    return tf.squeeze(rotated_points, axis=-1)


def gather_transform(batch_indices, points, rotation, translation):
    rotation_gather = tf.gather(rotation, batch_indices)
    translation_gather = tf.gather(translation, batch_indices)
    return rotate_points_3d(rotation_gather, points) + translation_gather


def gather_translate(batch_indices, points, translation):
    translation_gather = tf.gather(translation, batch_indices)
    return points + translation_gather
