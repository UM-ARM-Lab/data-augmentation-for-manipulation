#!/usr/bin/env python
import os
import pathlib
import pickle
import time
from collections import OrderedDict
from typing import Optional, Dict, List, Sequence

import git
import numpy as np
import tensorflow as tf
from colorama import Fore

import genpy
from arc_utilities.filesystem_utils import mkdir_and_ask
from link_bot_data.load_dataset import guess_dataset_size
from link_bot_data.ros_msg_serialization import ros_msg_to_bytes_tensor, bytes_to_ros_msg
from link_bot_pycommon import pycommon
from link_bot_pycommon.grid_utils import pad_voxel_grid
from link_bot_pycommon.serialization import dump_gzipped_pickle
from moonshine.filepath_tools import load_params
from moonshine.moonshine_utils import remove_batch, add_batch, numpify, to_list_of_strings
from moveit_msgs.msg import PlanningScene

NULL_PAD_VALUE = -10000

DEFAULT_VAL_SPLIT = 0.125
DEFAULT_TEST_SPLIT = 0.125

# FIXME this is hacky as hell
STRING_KEYS = [
    'tfrecord_path',
    'joint_names',
    'scene_msg',
]


def multigen(gen_func):
    """
    Use this as a decorator on a generator so that you can call it repeatedly
    Args:
        gen_func:

    Returns:

    """

    class _multigen:
        def __init__(self, *args, **kwargs):
            self.__args = args
            self.__kwargs = kwargs

        def __iter__(self):
            return gen_func(*self.__args, **self.__kwargs)

    return _multigen


def state_dict_is_null(state: Dict):
    for v in state.values():
        if np.any(v == NULL_PAD_VALUE):
            return True
    return False


def state_dict_is_null_tf(state: Dict):
    for v in state.values():
        if tf.reduce_any(tf.equal(v, NULL_PAD_VALUE)):
            return True
    return False


def total_state_dim(state: Dict):
    """
    :param state: assumed to be [batch, state_dim]
    :return:
    """
    state_dim = 0
    for v in state.values():
        state_dim += int(v.shape[1] / 2)
    return state_dim


def parse_and_deserialize(dataset, feature_description, n_parallel_calls=None):
    parsed_dataset = parse_dataset(dataset, feature_description, n_parallel_calls=n_parallel_calls)
    deserialized_dataset = deserialize(parsed_dataset, n_parallel_calls=n_parallel_calls)
    return deserialized_dataset


def parse_and_slow_deserialize(dataset, feature_description, n_parallel_calls=None):
    parsed_dataset = parse_dataset(dataset, feature_description, n_parallel_calls=n_parallel_calls)
    deserialized_dataset = slow_deserialize(parsed_dataset, n_parallel_calls=n_parallel_calls)
    return deserialized_dataset


def parse_dataset(dataset, feature_description, n_parallel_calls=None):
    def _parse(example_proto):
        deserialized_dict = tf.io.parse_single_example(example_proto, feature_description)
        return deserialized_dict

    # the elements of parsed dataset are dictionaries with the serialized tensors as strings
    parsed_dataset = dataset.map(_parse, num_parallel_calls=n_parallel_calls)
    return parsed_dataset


def slow_deserialize(parsed_dataset: tf.data.Dataset, n_parallel_calls=None):
    def _slow_deserialize(serialized_dict):
        deserialized_dict = {}
        for _key, _serialized_tensor in serialized_dict.items():
            def _parse(_t, types):
                for type in types:
                    try:
                        _deserialized_tensor = tf.io.parse_tensor(_t, type)
                        return _deserialized_tensor
                    except Exception:
                        pass
                raise ValueError("could not match to any of the given types")

            tf_types = [tf.float32, tf.float64, tf.int32, tf.int64, tf.string]
            deserialized_dict[_key] = _parse(_serialized_tensor, tf_types)
        return deserialized_dict

    for e in parsed_dataset:
        yield _slow_deserialize(e)


def deserialize(parsed_dataset: tf.data.Dataset, n_parallel_calls=None):
    def _deserialize(serialized_dict):
        deserialized_dict = {}
        for _key, _serialized_tensor in serialized_dict.items():
            if _key in STRING_KEYS:
                _deserialized_tensor = tf.io.parse_tensor(_serialized_tensor, tf.string)
            else:
                _deserialized_tensor = tf.io.parse_tensor(_serialized_tensor, tf.float32)
            deserialized_dict[_key] = _deserialized_tensor
        return deserialized_dict

    deserialized_dataset = parsed_dataset.map(_deserialize, num_parallel_calls=n_parallel_calls)
    return deserialized_dataset


def infer_shapes(parsed_dataset: tf.data.Dataset):
    element = next(iter(parsed_dataset))
    inferred_shapes = {}
    for key, serialized_tensor in element.items():
        if key in STRING_KEYS:
            inferred_shapes[key] = ()
        else:
            deserialized_tensor = tf.io.parse_tensor(serialized_tensor, tf.float32)
            inferred_shapes[key] = deserialized_tensor.shape
    return inferred_shapes


def dict_of_float_tensors_to_bytes_feature(d):
    return {k: float_tensor_to_bytes_feature(v) for k, v in d.items()}


def ros_msg_to_bytes_feature(msg):
    bt = ros_msg_to_bytes_tensor(msg)
    st = tf.io.serialize_tensor(bt)
    return bytes_feature(st.numpy())


def generic_to_bytes_feature(value):
    v = tf.convert_to_tensor(value)
    return bytes_feature(tf.io.serialize_tensor(v).numpy())


def float_tensor_to_bytes_feature(value):
    return bytes_feature(tf.io.serialize_tensor(tf.convert_to_tensor(value, dtype=tf.float32)).numpy())


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int_feature(values):
    """Returns a int64 from 1-dimensional numpy array"""
    assert values.ndim == 1
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def float_feature(values):
    """Returns a float_list from 1-dimensional numpy array"""
    assert values.ndim == 1
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def get_batch_dim(x):
    return tf.cast(x.shape[0], tf.int64)


def add_batch_map(example: Dict):
    batch_size = tf.numpy_function(get_batch_dim, [example['traj_idx']], tf.int64)
    example['batch_size'] = batch_size
    return example


def batch_tf_dataset(dataset, batch_size: int, drop_remainder: bool = True):
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    if isinstance(dataset, tf.data.Dataset):
        dataset = dataset.map(add_batch_map)
    return dataset


def filter_and_cache(dataset, filter_func):
    dataset = dataset.filter(filter_func)
    dataset = dataset.cache(cachename())
    return dataset


def cachename(mode: Optional[str] = None):
    if 'TF_CACHE_ROOT' in os.environ:
        cache_root = pathlib.Path(os.environ['TF_CACHE_ROOT'])
        cache_root.mkdir(exist_ok=True, parents=True)
    else:
        cache_root = pathlib.Path('/tmp')
    if mode is not None:
        tmpname = cache_root / f"{mode}_{pycommon.rand_str()}"
    else:
        tmpname = cache_root / f"{pycommon.rand_str()}"
    return str(tmpname)


def git_sha():
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha[:10]
    return sha


def make_unique_outdir(outdir: pathlib.Path, *names):
    now = str(int(time.time()))
    sha = git_sha()
    format_string = "{}_{}_{}" + len(names) * '_{}'
    full_output_directory = pathlib.Path(format_string.format(outdir, now, sha, *names))
    if outdir:
        if full_output_directory.is_file():
            print(Fore.RED + "argument outdir is an existing file, aborting." + Fore.RESET)
            return
        elif not full_output_directory.is_dir():
            mkdir_and_ask(full_output_directory, parents=True)
    return full_output_directory


NEW_PREFIX = 'new/'
PREDICTED_PREFIX = 'predicted/'
POSITIVE_PREFIX = 'positive/'
NEXT_PREFIX = 'next/'


def remove_predicted(k: str):
    if k.startswith(PREDICTED_PREFIX):
        return k[len(PREDICTED_PREFIX):]
    else:
        return k


def replaced_true_with_predicted(d: Dict):
    keys_to_pop = []
    out_d = d.copy()
    for k in out_d.keys():
        k_predicted_removed = remove_predicted(k)
        if k.startswith(PREDICTED_PREFIX) and k_predicted_removed in out_d:
            keys_to_pop.append(k_predicted_removed)
    for k in keys_to_pop:
        out_d.pop(k)
    return {remove_predicted(k): v for k, v in out_d.items()}


def add_positive(feature_name):
    return POSITIVE_PREFIX + feature_name


def add_next(feature_name):
    return NEXT_PREFIX + feature_name


def add_predicted(feature_name: str):
    return PREDICTED_PREFIX + feature_name


def add_new(feature_name: str):
    return NEW_PREFIX + feature_name


def null_pad(sequence, start=None, end=None):
    if isinstance(sequence, tf.Tensor):
        new_sequence = sequence.numpy().copy()
    else:
        new_sequence = sequence.copy()
    if start is not None and start > 0:
        new_sequence[:start] = NULL_PAD_VALUE
    if end is not None and end + 1 < len(sequence):
        new_sequence[end + 1:] = NULL_PAD_VALUE
    return new_sequence


def is_reconverging(labels, label_threshold=0.5):
    """
    :param labels: a [B, H] matrix of 1/0 or true/false
    :param label_threshold: numbers above this threshold are considered true
    :return: a [B] binary vector with [i] true for if labels[i] is reconverging
    """
    float_labels = tf.cast(labels, tf.float32)
    int_labels = tf.cast(labels, tf.int64)
    starts_with_1 = float_labels[:, 0] > label_threshold
    ends_with_1 = float_labels[:, -1] > label_threshold
    num_ones = tf.reduce_sum(int_labels, axis=1)
    index_of_last_1 = float_labels.shape[1] - tf.argmax(tf.reverse(float_labels, axis=[1]), axis=1) - 1
    reconverging = (index_of_last_1 >= num_ones)
    reconverging_and_start_end_1 = tf.stack([reconverging, starts_with_1, ends_with_1], axis=1)
    return tf.reduce_all(reconverging_and_start_end_1, axis=1)


def num_reconverging(labels):
    """
    :param labels: [B, H] matrix
    :return:
    """

    return tf.math.reduce_sum(tf.cast(is_reconverging(labels), dtype=tf.int32))


def num_reconverging_subsequences(labels):
    """
    :param labels: [B, H] matrix
    :return:
    """
    n = 0
    for start_idx in range(labels.shape[1]):
        for end_idx in range(start_idx + 2, labels.shape[1] + 1):
            n_i = num_reconverging(labels[:, start_idx:end_idx])
            n += n_i
    return n


def filter_only_reconverging(example):
    is_close = example['is_close']
    return remove_batch(is_reconverging(add_batch(is_close)))


def filter_no_reconverging(example):
    is_close = example['is_close']
    return tf.logical_not(remove_batch(is_reconverging(add_batch(is_close))))


def get_maybe_predicted(e: Dict, k: str):
    if k in e and add_predicted(k) in e:
        raise ValueError(f"ambiguous, dict has both {k} and {add_predicted(k)}")
    elif not (k in e or add_predicted(k) in e):
        raise ValueError(f"dict lacks both {k} and {add_predicted(k)}")
    elif k in e:
        return e[k]
    elif add_predicted(k) in e:
        return e[add_predicted(k)]
    else:
        raise RuntimeError()


def in_maybe_predicted(k: str, e: Dict):
    if k in e and add_predicted(k) in e:
        raise ValueError(f"ambiguous, dict has both {k} and {add_predicted(k)}")
    elif not (k in e or add_predicted(k) in e):
        return False
    return True


def use_gt_rope(example: Dict):
    if 'gt_rope' in example:
        example['rope'] = example['gt_rope']
    return example


def add_label(example: Dict, threshold: float):
    is_close = example['error'] < threshold
    example['is_close'] = tf.cast(is_close, dtype=tf.float32)


def coerce_types(d: Dict):
    """
    Converts the types of things in the dict to whatever we want for saving it
    Args:
        d:

    Returns:

    """
    out_d = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            if v.dtype == np.float64:
                out_d[k] = v.astype(np.float32)
            else:
                out_d[k] = v
        elif isinstance(v, np.float64):
            out_d[k] = np.float32(v)
        elif isinstance(v, np.float32):
            out_d[k] = v
        elif isinstance(v, np.int64):
            out_d[k] = v
        elif isinstance(v, tf.Tensor):
            v = v.numpy()
            if isinstance(v, np.ndarray):
                if v.dtype == np.float64:
                    out_d[k] = v.astype(np.float32)
                else:
                    out_d[k] = v
            elif isinstance(v, np.float64):
                out_d[k] = np.float32(v)
            else:
                out_d[k] = v
        elif isinstance(v, genpy.Message):
            out_d[k] = v
        elif isinstance(v, str):
            out_d[k] = v
        elif isinstance(v, int):
            out_d[k] = v
        elif isinstance(v, float):
            out_d[k] = v
        elif isinstance(v, list):
            v0 = v[0]
            if isinstance(v0, int):
                out_d[k] = np.array(v)
            elif isinstance(v0, float):
                out_d[k] = np.array(v, dtype=np.float32)
            elif isinstance(v0, genpy.Message):
                out_d[k] = np.array(v, dtype=object)
            elif isinstance(v0, str):
                out_d[k] = v
            elif isinstance(v0, list):
                v00 = v0[0]
                if isinstance(v00, int):
                    out_d[k] = np.array(v)
                elif isinstance(v00, float):
                    out_d[k] = np.array(v, dtype=np.float32)
                elif isinstance(v00, str):
                    out_d[k] = v
                elif isinstance(v00, genpy.Message):
                    out_d[k] = np.array(v, dtype=object)
            elif isinstance(v0, np.ndarray):
                if v0.dtype == np.float64:
                    out_d[k] = np.array(v).astype(np.float32)
                else:
                    out_d[k] = np.array(v)
            else:
                raise NotImplementedError(f"{k} {type(v)} {v}")
        elif isinstance(v, dict):
            out_d[k] = coerce_types(v)
        else:
            raise NotImplementedError(f"{k} {type(v)}")
    assert len(out_d) == len(d)
    return out_d


def pkl_write_example(full_output_directory, example, traj_idx, extra_metadata_keys: Optional[List[str]] = None):
    example_filename = index_to_filename('.pkl.gz', traj_idx)

    if 'metadata' in example:
        metadata = example.pop('metadata')
    else:
        metadata = {}
    metadata['data'] = example_filename
    if extra_metadata_keys is not None:
        for k in extra_metadata_keys:
            metadata[k] = example.pop(k)
    metadata_filename = index_to_filename('.pkl', traj_idx)
    full_metadata_filename = full_output_directory / metadata_filename

    metadata = coerce_types(metadata)
    with full_metadata_filename.open("wb") as metadata_file:
        pickle.dump(metadata, metadata_file)

    full_example_filename = full_output_directory / example_filename
    example = coerce_types(example)
    dump_gzipped_pickle(example, full_example_filename)

    return full_example_filename, full_metadata_filename


def tf_write_example(full_output_directory: pathlib.Path,
                     example: Dict,
                     example_idx: int):
    features = convert_to_tf_features(example)
    return tf_write_features(full_output_directory, features, example_idx)


def tf_write_features(full_output_directory: pathlib.Path, features: Dict, example_idx: int):
    record_filename = index_to_record_name(example_idx)
    full_filename = full_output_directory / record_filename
    features['tfrecord_path'] = bytes_feature(full_filename.as_posix().encode("utf-8"))
    example_proto = tf.train.Example(features=tf.train.Features(feature=features))
    example_str = example_proto.SerializeToString()
    record_options = tf.io.TFRecordOptions(compression_type='ZLIB')
    with tf.io.TFRecordWriter(str(full_filename), record_options) as writer:
        writer.write(example_str)
    return full_filename


def count_up_to_next_record_idx(full_output_directory):
    record_idx = 0
    while True:
        record_filename = index_to_record_name(record_idx)
        full_filename = full_output_directory / record_filename
        if not full_filename.exists():
            break
        record_idx += 1
    return record_idx


def convert_to_tf_features(example: Dict):
    features = {}
    for k, v in example.items():
        if isinstance(v, genpy.Message):
            f = ros_msg_to_bytes_feature(v)
        else:
            f = generic_to_bytes_feature(v)
        features[k] = f
    return features


class FilterConditional:

    def __init__(self, threshold: float, comparator):
        self.threshold = threshold
        self.comparator = comparator

    def __call__(self, x):
        return x.__getattribute__(self.comparator)(self.threshold)


def get_filter(name: str, **kwargs):
    filter_description = kwargs.get(name, None)
    if filter_description is not None:
        threshold = float(filter_description[1:])
        if filter_description[0] == '>':
            comparator = '__gt__'
        elif filter_description[0] == '<':
            comparator = '__lt__'
        else:
            raise ValueError(f"invalid comparator {filter_description[0]}")
        return FilterConditional(threshold, comparator)

    def _always_true_filter(x):
        return True

    return _always_true_filter


def modify_pad_env(example: Dict, h, w, c):
    padded_env, new_origin, new_extent = pad_voxel_grid(voxel_grid=example['env'],
                                                        origin=example['origin'],
                                                        res=example['res'],
                                                        new_shape=[h, w, c])
    example['env'] = padded_env
    example['extent'] = new_extent
    example['origin'] = new_origin
    return example


def pprint_example(example):
    for k, v in example.items():
        if hasattr(v, 'shape'):
            print(k, v.shape)
        elif isinstance(v, OrderedDict):
            print(k, numpify(v))
        else:
            print(k, v)


def index_to_record_name(traj_idx):
    return index_to_filename('.tfrecords', traj_idx)


def index_to_filename(file_extension, traj_idx):
    new_filename = f"example_{traj_idx:08d}{file_extension}"
    return new_filename


def train_test_split_counts(n: int, val_split: int = DEFAULT_VAL_SPLIT, test_split: int = DEFAULT_TEST_SPLIT):
    n_test = int(test_split * n)
    n_val = int(val_split * n)
    n_train = n - n_test - n_val
    return n_train, n_val, n_test


def compute_batch_size_for_n_examples(total_examples: int, max_batch_size: int):
    batch_size = min(max(1, int(total_examples / 2)), max_batch_size)
    return batch_size


def compute_batch_size(dataset_dirs: List[pathlib.Path], max_batch_size: int):
    total_examples = 0
    for dataset_dir in dataset_dirs:
        # assumes validation is smaller than or the same size as train
        total_examples += guess_dataset_size(dataset_dir)
    return compute_batch_size_for_n_examples(total_examples, max_batch_size)


def merge_hparams_dicts(dataset_dirs, verbose: int = 0):
    out_hparams = {}
    for dataset_dir in dataset_dirs:
        hparams = load_params(dataset_dir)
        for k, v in hparams.items():
            if k not in out_hparams:
                out_hparams[k] = v
            elif hparams[k] == v:
                pass
            elif verbose >= 0:
                msg = "Datasets have differing values for the hparam {}, using value {}".format(k, hparams[k])
                print(Fore.RED + msg + Fore.RESET)
    return out_hparams


def batch_sequence(s: Sequence, n, drop_remainder: bool):
    original_length = len(s)
    if drop_remainder:
        l = int(original_length / n) * n
    else:
        l = original_length
    for ndx in range(0, l, n):
        yield s[ndx:ndx + n]


def write_example(full_output_directory: pathlib.Path,
                  example: Dict,
                  example_idx: int,
                  save_format: str,
                  extra_metadata_keys: Optional[List[str]] = None):
    if save_format == 'tfrecord':
        tf_write_example(full_output_directory, example, example_idx)
    elif save_format == 'pkl':
        pkl_write_example(full_output_directory, example, example_idx, extra_metadata_keys)
    else:
        raise NotImplementedError()


def label_is(label_is, key='is_close'):
    def __filter(example):
        result = tf.squeeze(tf.equal(example[key][1], label_is))
        return result

    return __filter


def deserialize_scene_msg(example: Dict):
    if 'scene_msg' in example:
        scene_msg = _deserialize_scene_msg(example)

        example['scene_msg'] = scene_msg


def _deserialize_scene_msg(example):
    scene_msg = example['scene_msg']
    if isinstance(scene_msg, tf.Tensor):
        scene_msg = scene_msg.numpy()

    if isinstance(scene_msg, np.ndarray):
        assert scene_msg.ndim == 1
        if not isinstance(scene_msg[0], PlanningScene):
            scene_msg = np.array([bytes_to_ros_msg(m_i, PlanningScene) for m_i in scene_msg])
    elif isinstance(scene_msg, bytes):
        scene_msg = bytes_to_ros_msg(scene_msg, PlanningScene)
        # FIXME: why when I deserialize is it sometimes a list of bytes and sometimes a list of strings?
        scene_msg.robot_state.joint_state.name = to_list_of_strings(scene_msg.robot_state.joint_state.name)
    return scene_msg