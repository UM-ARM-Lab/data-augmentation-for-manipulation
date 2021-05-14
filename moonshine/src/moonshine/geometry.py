import tensorflow as tf


def make_rotation_matrix_like(x, theta):
    # https://en.wikipedia.org/wiki/Rotation_matrix
    z = tf.zeros_like(theta)
    one = tf.ones_like(theta)
    rotation_matrix = tf.stack([tf.stack([tf.cos(theta), -tf.sin(theta), z], axis=-1),
                                tf.stack([tf.sin(theta), tf.cos(theta), z], axis=-1),
                                tf.stack([z, z, one], axis=-1)], axis=-2)
    return rotation_matrix


# TODO: write tests
def rotate_points_3d(rotation_matrix, points):
    """

    Args:
        rotation_matrix: [b1, b2, ..., b2, 3, 3]
        points: [b1, b2, ..., b2, 3]

    Returns:

    """
    rotated_points = tf.matmul(rotation_matrix, tf.expand_dims(points, axis=-1))
    return tf.squeeze(rotated_points, axis=-1)
