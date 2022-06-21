import pathlib
from typing import Optional, List, Callable, Dict

import numpy as np
import tensorflow as tf
from colorama import Fore

import genpy
from link_bot_data.ros_msg_serialization import ros_msg_to_bytes_tensor
from moonshine.moonshine_utils import get_common_dicts, sequence_of_dicts_to_dict_of_sequences


def check_numerics(x, msg: Optional[str] = "found infs or nans!"):
    if isinstance(x, list):
        for v in x:
            if tf.is_tensor(v) and v.dtype in [tf.bfloat16, tf.float16, tf.float32, tf.float64]:
                tf.debugging.check_numerics(v, msg)
    elif isinstance(x, dict):
        for v in x.values():
            if tf.is_tensor(v) and v.dtype in [tf.bfloat16, tf.float16, tf.float32, tf.float64]:
                tf.debugging.check_numerics(v, msg)
    elif tf.is_tensor(x) and x.dtype in [tf.bfloat16, tf.float16, tf.float32, tf.float64]:
        tf.debugging.check_numerics(x, msg)


def batch_examples_dicts(examples: List):
    e_check = get_common_dicts(examples)
    examples_batch = {}
    for k in e_check.keys():
        v_check = e_check[k]
        values = [example[k] for example in examples]
        if isinstance(v_check, dict):
            examples_batch[k] = batch_examples_dicts(values)
        elif isinstance(v_check, list):
            v_check0 = v_check[0]
            if isinstance(v_check0, genpy.Message):
                examples_batch[k] = values
            else:
                with tf.device('/CPU:0'):
                    examples_batch[k] = tf.convert_to_tensor(np.array(values), name='convert_batched')
        elif isinstance(v_check, genpy.Message):
            examples_batch[k] = values
        elif isinstance(v_check, tf.Tensor):
            with tf.device('/CPU:0'):
                examples_batch[k] = tf.stack(values, axis=0, name='convert_batched')
        else:
            with tf.device('/CPU:0'):
                examples_batch[k] = tf.convert_to_tensor(np.array(values), name='convert_batched')
    return examples_batch


def listify(x):
    def _listify(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        elif isinstance(x, tuple):
            return tuple(_listify(x_i) for x_i in x)
        elif isinstance(x, list):
            return [_listify(x_i) for x_i in x]
        elif isinstance(x, tf.Tensor):
            x_np = x.numpy()
            return _listify(x_np)
        elif isinstance(x, dict):
            return {k: _listify(v) for k, v in x.items()}
        elif isinstance(x, np.int64):
            return int(x)
        elif isinstance(x, np.int32):
            return int(x)
        elif isinstance(x, np.float64):
            return float(x)
        elif isinstance(x, np.float32):
            return float(x)
        elif isinstance(x, int):
            return x
        elif isinstance(x, float):
            return x
        elif isinstance(x, str):
            return x
        elif isinstance(x, pathlib.Path):
            return x.as_posix()
        else:
            raise NotImplementedError(type(x))

    if isinstance(x, np.ndarray):
        return x.tolist()
    elif isinstance(x, list):
        return [_listify(x_i) for x_i in x]
    elif isinstance(x, tf.Tensor):
        x_np = x.numpy()
        return _listify(x_np)
    elif isinstance(x, dict):
        return {k: _listify(v) for k, v in x.items()}
    elif isinstance(x, float):
        return [x]
    elif isinstance(x, int):
        return [x]
    else:
        raise NotImplementedError(type(x))


def dict_of_tensors_to_dict_of_numpy_arrays(tf_dict):
    np_dict = {}
    for k, v in tf_dict.items():
        np_dict[k] = v.numpy()
    return np_dict


def flatten_batch_and_time(d):
    # assumes each element in d is of shape [b, t, ...]
    return {k: tf.reshape(v, [-1] + v.shape.as_list()[2:]) for k, v in d.items()}


def flatten_after(x, axis: int = 0):
    """ [N1, N2, ...] -> [N1, ..., N[axis], -1] """
    new_shape = x.shape.as_list()[:axis + 1] + [-1]
    return tf.reshape(x, new_shape)


def reduce_mean_dict(dict):
    reduced_dict = {}
    for k, v in dict.items():
        reduced_dict[k] = tf.reduce_mean(tf.stack(v, axis=0))
    return reduced_dict


def restore_variables(classifier_checkpoint: pathlib.Path, **variables):
    """
    Args:
        complete_checkpoint:
        **variables: the names are what to restore fomr, the values are what to restore to
            for example, 'conv_layers=model.conv_layers' would load conv_layers from the given checkpoint
            and use it to initialize model.conv_layers

    Returns:

    """
    model_checkpoint = tf.train.Checkpoint(**variables)
    # "model" matches the name used in the checkpoint (see ckpt creation in model_runner.py, "model=self.model")
    # the saved checkpoint contains things other than the model, hence why we have this second Checkpoint
    complete_checkpoint = tf.train.Checkpoint(model=model_checkpoint)
    checkpoint_manager = tf.train.CheckpointManager(complete_checkpoint, classifier_checkpoint.as_posix(),
                                                    max_to_keep=1)
    status = complete_checkpoint.restore(checkpoint_manager.latest_checkpoint)
    status.expect_partial()
    status.assert_existing_objects_matched()
    assert checkpoint_manager.latest_checkpoint is not None
    print(Fore.MAGENTA + "Restored {}".format(checkpoint_manager.latest_checkpoint) + Fore.RESET)


def to_list_of_strings(x):
    if isinstance(x[0], bytes):
        return [n.decode("utf-8") for n in x]
    elif isinstance(x[0], str):
        return [str(n) for n in x]
    elif isinstance(x, tf.Tensor):
        return [n.decode("utf-8") for n in x.numpy()]
    else:
        raise NotImplementedError()


def debuggable_tf_function(func: Callable, debug: bool):
    @tf.function
    def _non_debug_func(*args, **kwargs):
        return func(*args, **kwargs)

    def _debug_func(*args, **kwargs):
        return func(*args, **kwargs)

    if debug:
        return _debug_func
    else:
        return _non_debug_func


def reduce_mean_no_nan(x, axis=-1):
    """

    Args:
        x: [b,n] or just [n]

    Returns:
        mean, or 0 if it's empty

    """
    return tf.math.divide_no_nan(tf.reduce_sum(x, axis=axis), x.shape[axis])


def possibly_none_concat(old, new, axis: int):
    if old is None:
        return new
    else:
        return tf.concat([old, new], axis=axis)


def gather_dict(d: Dict, indices, axis: int = 0):
    """
    :param d: a dictionary where each value is a tensor/array with the same dimension along 'axis'
    :param indices: a 1-d tensor/array/vector of ints describing which elements to include from d
    :param axis: the axis to gather along
    :return:
    """
    out_d = {}
    for k, v in d.items():
        if isinstance(v[0], genpy.Message):
            out_d[k] = np.take(v, indices, axis=axis)
        else:
            out_d[k] = tf.gather(v, indices, axis=axis)
    return out_d


def vector_to_dict(description: Dict, z):
    start_idx = 0
    d = {}
    for k, dim in description.items():
        indices = tf.range(start_idx, start_idx + dim)
        d[k] = tf.gather(z, indices, axis=-1)
        start_idx += dim
    return d




def dict_of_numpy_arrays_to_dict_of_tensors(np_dict, dtype=tf.float32):
    tf_dict = {}
    for k, v in np_dict.items():
        tf_dict[k] = tf.convert_to_tensor(v, dtype=dtype)
    return tf_dict


def sequence_of_dicts_to_dict_of_tensors(seq_of_dicts, axis=0):
    dict_of_seqs = sequence_of_dicts_to_dict_of_sequences(seq_of_dicts)
    return {k: tf.stack(v, axis) for k, v in dict_of_seqs.items()}


def repeat_tensor(v, repetitions, axis, new_axis):
    if np.isscalar(v):
        multiples = []
    elif isinstance(v, genpy.Message):
        v = ros_msg_to_bytes_tensor(v)
        multiples = []
    else:
        multiples = [1] * v.ndim

    if new_axis:
        multiples.insert(axis, repetitions)
        v = tf.expand_dims(v, axis=axis)
        return tf.tile(v, multiples)
    else:
        multiples[axis] *= repetitions
        return tf.tile(v, multiples)


def dict_of_sequences_to_sequence_of_dicts_tf(dict_of_seqs, time_axis=0):
    # FIXME: a common problem I have is that I have a dictionary of tensors, each with the same shape in the first M dimensions
    # and I want to get those shapes, but I don't care which key/value I use. Feels like I need a different datastructure here.
    seq_of_dicts = []
    # assumes all values in the dict have the same first dimension size (num time steps)
    T = list(dict_of_seqs.values())[0].shape[time_axis]
    for t in range(T):
        dict_t = {}
        for k, v in dict_of_seqs.items():
            dict_t[k] = tf.gather(v, t, axis=time_axis)
        seq_of_dicts.append(dict_t)

    return seq_of_dicts
