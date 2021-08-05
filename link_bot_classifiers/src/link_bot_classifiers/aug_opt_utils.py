from dataclasses import dataclass

import tensorflow as tf

import rospy


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