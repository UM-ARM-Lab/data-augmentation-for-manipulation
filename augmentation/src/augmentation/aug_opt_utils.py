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
    to_local_frame = tf.reduce_mean(obj_points, axis=-2)  # [b,m,T,1,3]
    to_local_frame = tf.reduce_mean(to_local_frame, axis=-2)  # [b,m,1,1,3]

    to_local_frame_moved = tf.where(tf.cast(moved_mask[..., None], tf.bool), to_local_frame, 0)
    to_local_frame_moved_sum = tf.reduce_sum(to_local_frame_moved, axis=-2)
    to_local_frame_moved_count = tf.reduce_sum(moved_mask, axis=-1, keepdims=True)
    to_local_frame_moved_mean = to_local_frame_moved_sum / to_local_frame_moved_count
    to_local_frame_moved_mean_expanded = to_local_frame_moved_mean[:, None, None, :]

    obj_points_local_frame = obj_points - to_local_frame_moved_mean_expanded  # [b, m_objects, T, n_points, 3]
    transformation_matrices_expanded = tf.expand_dims(transformation_matrices, axis=-3)
    obj_points_aug_local_frame = transform_points_3d(transformation_matrices_expanded, obj_points_local_frame)
    obj_points_aug = obj_points_aug_local_frame + to_local_frame_moved_mean_expanded  # [b, m_objects, T, n_points, 3]
    return obj_points_aug, to_local_frame_moved_mean


def check_env_constraints(attract_mask, min_dist, res):
    half_res_expanded = res[:, None, None] / 2
    attract_satisfied = tf.cast(min_dist < half_res_expanded, tf.float32)
    repel_satisfied = tf.cast(min_dist > half_res_expanded, tf.float32)
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
    dparams = tf.einsum('bmni,bmnij->bmnj', dpoint, dpoint_dparams)  # [b,m,n_points,6]
    dparams = tf.reduce_mean(dparams, axis=-2)
    return dparams
