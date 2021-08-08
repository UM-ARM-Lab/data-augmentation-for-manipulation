from dataclasses import dataclass

import tensorflow as tf

import rospy
from moonshine.geometry import transform_points_3d


def debug_aug():
    return rospy.get_param("DEBUG_AUG", False)


def debug_aug_sgd():
    return rospy.get_param("DEBUG_AUG_SGD", False)


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


def transformation_obj_points(obj_points, transformation_matrices):
    to_local_frame = tf.reduce_mean(obj_points, axis=1, keepdims=True)
    obj_points_local_frame = obj_points - to_local_frame
    obj_points_aug_local_frame = transform_points_3d(transformation_matrices[:, None],
                                                     obj_points_local_frame)
    obj_points_aug = obj_points_aug_local_frame + to_local_frame
    return obj_points_aug, to_local_frame