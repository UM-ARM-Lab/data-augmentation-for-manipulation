import os
import pathlib
import random
import signal
import string
import traceback
import warnings
from typing import Union, List, Callable, Optional, Tuple, Iterable, Dict

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


def approx_range_split_counts(n: int, splits: int):
    return [len(s) for s in approx_range_split(n, splits)]


def approx_range_split(n: int, splits: int):
    if splits > n:
        raise ValueError(f"splits {splits} > n {n}")
    return np.array_split(range(n), splits)


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


def try_make_dict_tf_float32(d):
    f32d = {}
    for k, s_k in d.items():
        try:
            s_k = tf.convert_to_tensor(s_k)
            f32d[k] = tf.cast(s_k, tf.float32)
        except Exception:
            f32d[k] = s_k
    return f32d


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
    out_d = {}
    for k, s_k in d.items():
        out_d[k] = s_k.astype(np.float32)
    return out_d


def longest_reconverging_subsequence(x):
    max_start_idx = 0
    max_end_idx = 0
    start_idx = 0
    max_consecutive_zeros = 0
    consecutive_zeros = 0
    for i, x_i in enumerate(x):
        if x_i:
            if consecutive_zeros > max_consecutive_zeros:
                max_consecutive_zeros = consecutive_zeros
                max_start_idx = start_idx
                max_end_idx = i
            consecutive_zeros = 0
        else:
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
            n_leading_ones = sum(x[i:start_of_zeros])
            if n_leading_ones > max_leading_ones:
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
    elif isinstance(model_dirs, pathlib.Path):
        return [model_dirs]
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


def deal_with_exceptions(how_to_handle: str,
                         function: Callable,
                         exception_callback: Optional[Callable] = None,
                         exceptions: Optional[Tuple] = None,
                         value_on_no_retry_exception=None,
                         print_exception: bool = False,
                         **kwargs):
    def _print_exception():
        if print_exception:
            print("Caught an exception!")
            traceback.print_exc()
            print("End of caught exception.")

    if exceptions is None:
        exceptions = (Exception,)

    if how_to_handle == 'raise':
        return function(**kwargs)
    else:
        for i in range(10):
            try:
                return function(**kwargs)
            except exceptions:
                if exception_callback is not None:
                    exception_callback()
                if i == 9:
                    traceback.print_exc()
                if how_to_handle == 'retry':
                    _print_exception()
                elif how_to_handle == 'catch':
                    _print_exception()
                    return value_on_no_retry_exception
        return value_on_no_retry_exception


def catch_timeout(seconds: int, func: Callable, *args, **kwargs):
    def _handle_timeout(signum, frame):
        raise TimeoutError()

    try:
        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.alarm(seconds)
        try:
            result = func(*args, **kwargs)
            return result, False
        finally:
            signal.alarm(0)
    except TimeoutError:
        print("Caught timeout!")
        return None, True


def retry_on_timeout(t: int, on_timeout: Optional[Callable], f: Callable, *args, **kwargs):
    """
    For generators
    Args:
        t: timeout in seconds
        f: a generator, any function with `yield` or `field from`
        on_timeout: callback used when timeouts happen

    Returns:

    """
    it = f(*args, **kwargs)
    while True:
        try:
            i, timed_out = catch_timeout(t, next, it)
            if timed_out:
                it = f(*args, **kwargs)  # reset the generator
                if on_timeout is not None:
                    on_timeout()
            else:
                yield i
        except StopIteration:
            return


def skip_on_timeout(t: int, on_timeout: Optional[Callable], f: Callable, *args, **kwargs):
    """
    For generators
    Args:
        t: timeout in seconds
        f: a generator, any function with `yield` or `field from`
        on_timeout: callback used when timeouts happen

    Returns:

    """
    it = f(*args, **kwargs)
    while True:
        try:
            i, timed_out = catch_timeout(t, next, it)
            if not timed_out:
                yield i
        except StopIteration:
            return


def quote_string(s: str):
    return f'f"{s}"'


def pathify(x):
    """
    Convert any strings that contain os.sep to pathlib.Path objects. works with arbitrarily nested data structures
    Args:
        x:

    Returns:

    """
    if isinstance(x, str):
        if os.sep in x:
            return pathlib.Path(x)
        else:
            return x
    elif isinstance(x, dict):
        return {k: pathify(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [pathify(v) for v in x]
    else:
        return x


def dgather(d: Dict, keys: Iterable[str]):
    return {k: d[k] for k in keys}


def densify_points(batch_size, points, num_densify=5):
    """
    Args:
        points: [b, n, 3]
    Returns: [b, n * num_density, 3]
    """
    if points.shape[1] <= 1:
        return points

    starts = points[:, :-1]
    ends = points[:, 1:]
    linspaced = tf.linspace(starts, ends, num_densify, axis=2)  # [b, n, num_density, 3]
    densitifed_points = tf.reshape(linspaced, [batch_size, -1, 3])
    return densitifed_points


def update_if_valid(d: Dict, is_valid, k: str, v_aug):
    d[k] = is_valid * v_aug + (1 - is_valid) * d[k]