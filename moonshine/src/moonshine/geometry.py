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
    dist = a_s - 2 * tf.matmul(a, b, transpose_b=True) + tf.linalg.matrix_transpose(b_s)  # [b, ..., n, m]
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


def transformation_params_to_matrices(params, batch_size):
    """

    Args:
        params:  [b,6] in the form [x,y,z,roll,pitch,yaw]
        batch_size: int, the value of b

    Returns: [b,4,4] with the assumption of roll, pitch, yaw, then translation (aka the normal thing)

    """
    translation = params[:, :3][:, :, None]
    angles = params[:, 3:]
    r33 = from_euler(angles)
    r34 = tf.concat([r33, translation], axis=2)
    h = tf.tile(tf.constant([[[0, 0, 0, 1]]], dtype=tf.float32), [batch_size, 1, 1])
    matrices = tf.concat([r34, h], axis=1)
    return matrices


# COPIED FROM TENSORFLOW-GRAPHICS
def from_euler(angles, name=None):
    r"""Convert an Euler angle representation to a rotation matrix.

    The resulting matrix is $$\mathbf{R} = \mathbf{R}_z\mathbf{R}_y\mathbf{R}_x$$.

    Note:
      In the following, A1 to An are optional batch dimensions.

    Args:
      angles: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
        represents the three Euler angles. `[A1, ..., An, 0]` is the angle about
        `x` in radians `[A1, ..., An, 1]` is the angle about `y` in radians and
        `[A1, ..., An, 2]` is the angle about `z` in radians.
      name: A name for this op that defaults to "rotation_matrix_3d_from_euler".

    Returns:
      A tensor of shape `[A1, ..., An, 3, 3]`, where the last two dimensions
      represent a 3d rotation matrix.

    Raises:
      ValueError: If the shape of `angles` is not supported.
    """
    sin_angles = tf.sin(angles)
    cos_angles = tf.cos(angles)
    return _build_matrix_from_sines_and_cosines(sin_angles, cos_angles)


def _build_matrix_from_sines_and_cosines(sin_angles, cos_angles):
    """Builds a rotation matrix from sines and cosines of Euler angles.

    Note:
      In the following, A1 to An are optional batch dimensions.

    Args:
      sin_angles: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
        represents the sine of the Euler angles.
      cos_angles: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
        represents the cosine of the Euler angles.

    Returns:
      A tensor of shape `[A1, ..., An, 3, 3]`, where the last two dimensions
      represent a 3d rotation matrix.
    """
    sx, sy, sz = tf.unstack(sin_angles, axis=-1)
    cx, cy, cz = tf.unstack(cos_angles, axis=-1)
    m00 = cy * cz
    m01 = (sx * sy * cz) - (cx * sz)
    m02 = (cx * sy * cz) + (sx * sz)
    m10 = cy * sz
    m11 = (sx * sy * sz) + (cx * cz)
    m12 = (cx * sy * sz) - (sx * cz)
    m20 = -sy
    m21 = sx * cy
    m22 = cx * cy
    matrix = tf.stack((m00, m01, m02,
                       m10, m11, m12,
                       m20, m21, m22),
                      axis=-1)
    output_shape = tf.concat((tf.shape(input=sin_angles)[:-1], (3, 3)), axis=-1)
    return tf.reshape(matrix, shape=output_shape)
