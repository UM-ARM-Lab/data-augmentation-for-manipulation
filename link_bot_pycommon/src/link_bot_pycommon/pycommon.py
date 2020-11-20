import pathlib
import random
import string
import warnings
from typing import List, Union

import numpy as np
import tensorflow as tf


def directions_3d(pitch, yaw):
    """
    pitch : [B, S, T]
    yaw : [B, S, T]
    """
    # implement me, and test generating recovery action dataset
    # should be much faster than before?
    c1 = tf.math.cos(pitch)
    s1 = tf.math.sin(pitch)
    c2 = tf.math.cos(yaw)
    s2 = tf.math.sin(yaw)
    directions = tf.stack([c1 * c2, c1 * s2, -s1], axis=-1)
    return directions


def default_if_none(x, default):
    return default if x is None else x


def yaw_diff(a, b):
    diff = a - b
    greater_indeces = np.argwhere(diff > np.pi)
    diff[greater_indeces] = diff[greater_indeces] - 2 * np.pi
    less_indeces = np.argwhere(diff < -np.pi)
    diff[less_indeces] = diff[less_indeces] + 2 * np.pi
    return diff


def state_cost(s, goal):
    return np.linalg.norm(s[0, 0:2] - goal[0, 0:2])


def wrap_angle(angles):
    """ https://stackoverflow.com/questions/15927755/opposite-of-numpy-unwrap """
    return (angles + np.pi) % (2 * np.pi) - np.pi


def angle_from_configuration(state):
    warnings.warn("invalid for multi link ropes", DeprecationWarning)
    v1 = np.array([state[4] - state[2], state[5] - state[3]])
    v2 = np.array([state[0] - state[2], state[1] - state[3]])
    return angle_2d(v1, v2)


def angle_2d(v1, v2):
    return np.math.atan2(np.linalg.det([v1, v2]), np.dot(v1, v2))


def batch_dot_tf(v1, v2):
    return tf.einsum('ij,ij->i', v1, v2)


def angle_2d_batch_tf(v1, v2):
    """
    :param v1: [batch, n]
    :param v2:  [batch, n]
    :return: [batch]
    """
    return tf.math.atan2(tf.linalg.det(tf.stack((v1, v2), axis=1)), batch_dot_tf(v1, v2))


def n_state_to_n_links(n_state: int):
    return int(n_state // 2 - 1)


def n_state_to_n_points(n_state: int):
    return int(n_state // 2)


def make_random_rope_configuration(extent, n_state, link_length, max_angle_rad, rng: np.random.RandomState):
    """
    First sample a head point, then sample angles for the other points
    :param max_angle_rad: NOTE, by sampling uniformly here we make certain assumptions about the planning task
    :param extent: bounds of the environment [xmin, xmax, ymin, ymax] (meters)
    :param link_length: length of each segment of the rope (meters)
    :return:
    """

    def oob(x, y):
        return not (extent[0] < x < extent[1] and extent[2] < y < extent[3])

    n_links = n_state_to_n_links(n_state)
    theta = rng.uniform(-np.pi, np.pi)
    valid = False
    while not valid:
        head_x = rng.uniform(extent[0], extent[1])
        head_y = rng.uniform(extent[2], extent[3])

        rope_configuration = np.zeros(n_state)
        rope_configuration[-2] = head_x
        rope_configuration[-1] = head_y

        j = n_state - 1
        valid = True
        for i in range(n_links):
            theta = theta + rng.uniform(-max_angle_rad, max_angle_rad)
            rope_configuration[j - 2] = rope_configuration[j] + np.cos(theta) * link_length
            rope_configuration[j - 3] = rope_configuration[j - 1] + np.sin(theta) * link_length

            if oob(rope_configuration[j - 2], rope_configuration[j - 3]):
                valid = False
                break

            j = j - 2

    return rope_configuration


def flatten_points(points):
    return np.array([[p.x, p.y] for p in points]).flatten()


def flatten_named_points(points):
    return np.array([[p.point.x, p.point.y] for p in points]).flatten()


def transpose_2d_lists(l):
    # https://stackoverflow.com/questions/6473679/transpose-list-of-lists
    return list(map(list, zip(*l)))


def print_dict(example):
    for k, v in example.items():
        if hasattr(v, 'dtype'):
            dtype = v.dtype
        else:
            dtype = type(v)
        if hasattr(v, 'shape'):
            shape = v.shape
        else:
            shape = '?'
        print(f"{k:30s} {str(dtype):20s} {str(shape)}")


def rand_str(length=16):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


def vector_to_points_2d(x):
    x_points = x.reshape(-1, 2)
    xs = x_points[:, 0]
    ys = x_points[:, 1]
    return xs, ys


def make_dict_tf_float32(d):
    f32d = {}
    for k, s_k in d.items():
        s_k = tf.convert_to_tensor(s_k)
        if s_k.dtype == tf.string:
            f32d[k] = s_k
        else:
            f32d[k] = tf.cast(s_k, tf.float32)
    return f32d


def make_dict_float32(d):
    return {k: s_k.astype(np.float32) for k, s_k in d.items()}


def longest_reconverging_subsequence(x):
    max_start_idx = 0
    max_end_idx = 0
    start_idx = 0
    max_consecutive_zeros = 0
    consecutive_zeros = 0
    for i, x_i in enumerate(x):
        if x_i == 1:
            if consecutive_zeros > max_consecutive_zeros:
                max_consecutive_zeros = consecutive_zeros
                max_start_idx = start_idx
                max_end_idx = i
            consecutive_zeros = 0
        if x_i == 0:
            if consecutive_zeros == 0:
                start_idx = i
            consecutive_zeros += 1
    return max_start_idx, max_end_idx


def trim_reconverging(x, max_leading_ones=3, max_trailing_ones=3):
    start_of_zeros, end_of_zeros = longest_reconverging_subsequence(x)
    assert start_of_zeros != 0

    # expand start index
    if start_of_zeros == 1:
        just_before_start_of_zeros = 0
    else:
        just_before_start_of_zeros = 0
        for i in range(start_of_zeros - 1, -1, -1):
            if start_of_zeros - just_before_start_of_zeros > max_leading_ones:
                break
            if x[i] == 0:
                just_before_start_of_zeros = i + 1
                break

    # expand end index
    if end_of_zeros == len(x):
        end_of_ones_after_zeros = end_of_zeros
    else:
        end_of_ones_after_zeros = end_of_zeros
        for i, x_i in enumerate(x[end_of_zeros:]):
            if x_i - end_of_zeros > max_leading_ones:
                break
            if x_i == 0:
                break
            end_of_ones_after_zeros += 1

    return just_before_start_of_zeros, end_of_ones_after_zeros


def paths_from_json(model_dirs):
    if isinstance(model_dirs, list):
        return [pathlib.Path(s) for s in model_dirs]
    elif isinstance(model_dirs, str):
        return [pathlib.Path(model_dirs)]
    elif model_dirs is None:
        return None
    else:
        raise NotImplementedError(type(model_dirs))


def paths_to_json(model_dirs: Union[List[pathlib.Path], pathlib.Path]) -> Union[List[str], str, None]:
    if isinstance(model_dirs, list):
        return [p.as_posix() for p in model_dirs]
    elif isinstance(model_dirs, pathlib.Path):
        return model_dirs.as_posix()
    elif isinstance(model_dirs, str):
        return model_dirs
    elif model_dirs is None:
        return None
    else:
        raise NotImplementedError()


def log_scale_0_to_1(x, k=10):
    """
    Performs a log rescaling of the numbers from 0 to 1
    0 still maps to 0 and 1 still maps to 1, but the numbers get squished
    so that small values are larger. k controls the amount of squishedness,
    larger is more squished
    """
    return np.log(k * x + 1) / np.log(k + 1)
