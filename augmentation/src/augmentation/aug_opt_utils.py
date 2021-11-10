from dataclasses import dataclass

import tensorflow as tf

import rospy
from moonshine.geometry import transform_points_3d


def debug_aug():
    return rospy.get_param("DEBUG_AUG", False)


def debug_input():
    return rospy.get_param("DEBUG_INPUT", False)


def debug_ik():
    return rospy.get_param("DEBUG_IK", False)


@dataclass
class MinDists:
    attract: tf.Tensor
    repel: tf.Tensor
    robot_repel: tf.Tensor


@dataclass
class EnvPoints:
    full: tf.Tensor
    sparse: tf.Tensor


@dataclass
class EnvOptDebugVars:
    nearest_attract_env_points: tf.Tensor
    nearest_repel_points: tf.Tensor
    nearest_robot_repel_points: tf.Tensor


def subsample_points(points, fraction):
    """

    Args:
        points: [n, 3]
        fraction: from 0.0 to 1.0

    Returns:

    """
    n_take_every = int(1 / fraction)
    return points[::n_take_every]


def transform_obj_points(obj_points, moved_mask, transformation_matrices):
    """

    Args:
        obj_points: [b,m,T,n_points,3]
        moved_mask: [b,m,...]
        transformation_matrices: [b,k,4,4]
            considered to in the frame of the obj_points,
            which is defined as the same orientation as the world but with the position being the center of obj_points

    Returns: [b,k,T,n_points,3], [b,3]

    """
    to_local_frame = tf.reduce_mean(obj_points, axis=-2)  # [b,m,T,3]
    to_local_frame = tf.reduce_mean(to_local_frame, axis=-2)  # [b,m,3]

    to_local_frame_moved_mean = mean_over_moved(moved_mask, to_local_frame)  # [b, 3]
    to_local_frame_moved_mean_expanded = to_local_frame_moved_mean[:, None, None, None, :]

    obj_points_local_frame = obj_points - to_local_frame_moved_mean_expanded  # [b, m_objects, T, n_points, 3]
    transformation_matrices_expanded = transformation_matrices[..., None, None, :, :, :]
    obj_points_aug_local_frame = transform_points_3d(transformation_matrices_expanded, obj_points_local_frame)
    obj_points_aug = obj_points_aug_local_frame + to_local_frame_moved_mean_expanded  # [b, m_objects, T, n_points, 3]
    return obj_points_aug, to_local_frame_moved_mean


def sum_over_moved(moved_mask, x):
    """
    Gives the sum of x over dimension 1, which represent different objects, but the non-moved objects aren't included.

    Args:
        moved_mask: [b, m]
        x: [b, m, d1, ..., dn]

    Returns:

    """
    # replacing the values where moved_mask is false with zero will not affect the sum...
    moved_mask_expanded = expand_to_match(moved_mask, x)
    x_moved = tf.where(tf.cast(moved_mask_expanded, tf.bool), x, 0)
    x_moved_sum = tf.reduce_sum(x_moved, axis=1)
    return x_moved_sum


def mean_over_moved(moved_mask, x):
    """
    Gives the mean of x over dimension 1, which represent different objects, but the non-moved objects aren't included.

    Args:
        moved_mask: [b, m]
        x: [b, m, d1, ..., dn]

    Returns:

    """
    # replacing the values where moved_mask is false with zero will not affect the sum...
    x_moved_sum = sum_over_moved(moved_mask, x)
    # ... if we divide by the right numbers
    moved_count = tf.reduce_sum(moved_mask, axis=1)
    moved_count = expand_to_match(moved_count, x)
    x_moved_mean = x_moved_sum / moved_count
    return x_moved_mean


def expand_to_match(a, b):
    """

    Args:
        a: [b1, b2, ..., bN]
        b: [b1, b2, ..., bN, d1, d2, ... dN]

    Returns: a but with the shape [b1, b2, ..., bN, 1, 1, ... 1]

    """
    a_expanded = a
    for dim_j in range(b.ndim - 2):
        a_expanded = tf.expand_dims(a_expanded, axis=-1)
    return a_expanded


def check_env_constraints(attract_mask, min_dist):
    # we don't need to worry about equality because the (discrete) sdf is never exactly 0
    attract_satisfied = tf.cast(min_dist < 0, tf.float32)
    repel_satisfied = tf.cast(min_dist > 0, tf.float32)
    constraints_satisfied = (attract_mask * attract_satisfied) + ((1 - attract_mask) * repel_satisfied)
    return constraints_satisfied


def pick_best_params(invariance_model, sampled_params, batch_size):
    predicted_errors = invariance_model.evaluate(sampled_params)
    _, best_indices_all = tf.math.top_k(-predicted_errors, tf.cast(batch_size, tf.int32), sorted=False)
    best_indices_shuffled = tf.random.shuffle(best_indices_all, seed=0)
    best_indices = best_indices_shuffled[:batch_size]
    best_params = tf.gather(sampled_params, best_indices, axis=0)
    return best_params


def delta_min_dist_loss(sdf_dist, sdf_dist_aug):
    min_dist = tf.reduce_min(sdf_dist, axis=1)
    min_dist_aug = tf.reduce_min(sdf_dist_aug, axis=1)
    delta_min_dist = tf.abs(min_dist - min_dist_aug)
    return delta_min_dist


def dpoint_to_dparams(dpoint, dpoint_dparams):
    """

    Args:
        dpoint:  [b,m,T,n_points,3]
        dpoint_dparams: [b,m,T,n_points,3,p]

    Returns: [b,m,T,n_points,p]

    """
    return tf.squeeze(tf.matmul(tf.expand_dims(dpoint, -2), dpoint_dparams), axis=-2)
