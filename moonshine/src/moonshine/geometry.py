import numpy as np
import tensorflow as tf


def transform_points_3d(transform_matrix, points):
    """

    Args:
        transform_matrix: [b1, b2, ..., b2, 4, 4]
        points: [b1, b2, ..., b2, 3]

    Returns:

    """
    points_homo = tf.concat([points, tf.ones_like(points[..., 0:1])], axis=-1)
    points_homo = tf.expand_dims(points_homo, axis=-1)
    transformed_points = tf.matmul(transform_matrix, points_homo)
    return tf.squeeze(transformed_points, axis=-1)[..., :3]


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


def pairwise_squared_distances(a, b):
    """
    Adapted from https://github.com/ClayFlannigan/icp
    Computes pairwise distances between to sets of points

    Args:
        a: [b, ..., n, 3]
        b:  [b, ..., m, 3]

    Returns: [b, ..., n, m]

    """
    a_s = tf.reduce_sum(tf.square(a), axis=-1, keepdims=True)  # [b, ..., n, 1]
    b_s = tf.reduce_sum(tf.square(b), axis=-1, keepdims=True)  # [b, ..., m, 1]
    dist = a_s - 2 * tf.matmul(a, b, transpose_b=True) + tf.transpose(b_s)  # [b, ..., n, m]
    return dist


def best_fit_transform(a, b):
    """
    Adapted from https://github.com/ClayFlannigan/icp
    Calculates the least-squares best-fit transform that maps corresponding points a to b in m spatial dimensions
    Input:
      a: Nxm numpy array of corresponding points
      b: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps a on to b
      R: mxm rotation matrix
      t: mx1 translation vector
    """
    # get number of dimensions
    m = a.shape[1]

    # translate points to their centroids
    centroid_a = tf.reduce_mean(a, axis=0)
    centroid_b = tf.reduce_mean(b, axis=0)
    aa = a - centroid_a
    bb = b - centroid_b

    # rotation matrix
    h = tf.matmul(tf.transpose(aa, [1, 0]), bb)
    s, u, v = tf.linalg.svd(h)
    rotation = tf.matmul(tf.transpose(v, [1, 0]), tf.transpose(u, [1, 0]))

    # special reflection case
    if tf.linalg.det(rotation) < 0:
        v[m - 1, :] *= -1
        rotation = tf.matmul(tf.transpose(v, [1, 0]), tf.transpose(u, [1, 0]))

    # translation
    translation = tf.expand_dims(centroid_b, 1) - tf.matmul(rotation, tf.expand_dims(centroid_a, 1))

    return rotation, translation


def best_fit_translation(a, b):
    """
    Best fit translation that moves a to b
    Args:
        a: [b, ..., n, k], where k is usually 2 or 3
        b: [b, ..., n, k]

    Returns: [b, ..., k]

    """
    translation = tf.reduce_mean(b - a, axis=-2)
    return translation


def transform_dict_of_points_vectors(m: np.ndarray, d, keys):
    d_out = {}
    for k in keys:
        points = np.reshape(d[k], [-1, 3, 1])
        points_homo = np.concatenate([points, np.ones([points.shape[0], 1, 1])], axis=1)
        points_aug = np.matmul(m[None], points_homo)[:, :3, 0]
        d_out[k] = np.reshape(points_aug, -1).astype(np.float32)
    return d_out
