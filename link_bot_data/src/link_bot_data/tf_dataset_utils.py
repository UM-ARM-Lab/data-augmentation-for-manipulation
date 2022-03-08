import pathlib
import pickle
from typing import Dict, Optional, List

import numpy as np
import tensorflow as tf

import genpy
import rospy
from link_bot_data.coerce_types import coerce_types
from link_bot_data.dataset_utils import NULL_PAD_VALUE
from link_bot_data.ros_msg_serialization import ros_msg_to_bytes_tensor, bytes_to_ros_msg
from link_bot_pycommon.serialization import dump_gzipped_pickle
from moonshine.torch_and_tf_utils import remove_batch, add_batch
from moonshine.tensorflow_utils import to_list_of_strings
from moveit_msgs.msg import PlanningScene


def state_dict_is_null_tf(state: Dict):
    for v in state.values():
        if tf.reduce_any(tf.equal(v, NULL_PAD_VALUE)):
            return True
    return False


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


def tf_write_example(full_output_directory: pathlib.Path,
                     example: Dict,
                     example_idx: int):
    features = convert_to_tf_features(example)
    return tf_write_features(full_output_directory, features, example_idx)


def filter_no_reconverging(example):
    is_close = example['is_close']
    return tf.logical_not(remove_batch(is_reconverging(add_batch(is_close))))


add_label_has_printed = False


def add_label(example: Dict, threshold: float):
    global add_label_has_printed
    if not add_label_has_printed:
        print(f"Using threshold {threshold}")
        add_label_has_printed = True
    is_close = example['error'] < threshold
    example['is_close'] = tf.cast(is_close, dtype=tf.float32)


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


STRING_KEYS = [
    'tfrecord_path',
    'joint_names',
    'scene_msg',
]


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


def parse_and_deserialize(dataset, feature_description, n_parallel_calls=None):
    parsed_dataset = parse_dataset(dataset, feature_description, n_parallel_calls=n_parallel_calls)
    deserialized_dataset = deserialize(parsed_dataset, n_parallel_calls=n_parallel_calls)
    return deserialized_dataset


def parse_and_slow_deserialize(dataset, feature_description, n_parallel_calls=None):
    parsed_dataset = parse_dataset(dataset, feature_description, n_parallel_calls=n_parallel_calls)
    deserialized_dataset = slow_deserialize(parsed_dataset, n_parallel_calls=n_parallel_calls)
    return deserialized_dataset


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


def convert_to_tf_features(example: Dict):
    features = {}
    for k, v in example.items():
        if isinstance(v, genpy.Message):
            f = ros_msg_to_bytes_feature(v)
        else:
            f = generic_to_bytes_feature(v)
        features[k] = f
    return features


def write_example(full_output_directory: pathlib.Path,
                  example: Dict,
                  example_idx: int,
                  save_format: str,
                  extra_metadata_keys: Optional[List[str]] = None):
    if save_format == 'tfrecord':
        return tf_write_example(full_output_directory, example, example_idx)
    elif save_format == 'pkl':
        return pkl_write_example(full_output_directory, example, example_idx, extra_metadata_keys)
    else:
        raise NotImplementedError()


def pkl_write_example(full_output_directory, example, traj_idx, extra_metadata_keys: Optional[List[str]] = None):
    metadata_filename = index_to_filename('.pkl', traj_idx)
    full_metadata_filename = full_output_directory / metadata_filename
    example_filename = index_to_filename('.pkl.gz', traj_idx)
    full_example_filename = full_output_directory / example_filename

    if 'metadata' in example:
        metadata = example.pop('metadata')
    else:
        metadata = {}
    metadata['data'] = example_filename
    if extra_metadata_keys is not None:
        for k in extra_metadata_keys:
            metadata[k] = example.pop(k)

    metadata = coerce_types(metadata)
    with full_metadata_filename.open("wb") as metadata_file:
        pickle.dump(metadata, metadata_file)

    example = coerce_types(example)
    dump_gzipped_pickle(example, full_example_filename)

    return full_example_filename, full_metadata_filename


def index_to_filename(file_extension, traj_idx):
    new_filename = f"example_{traj_idx:08d}{file_extension}"
    return new_filename


def index_to_record_name(traj_idx):
    return index_to_filename('.tfrecords', traj_idx)


def index_to_filename2(traj_idx, save_format):
    if save_format == 'pkl':
        return index_to_filename('.pkl', traj_idx)
    elif save_format == 'tfrecord':
        return index_to_record_name(traj_idx)


def count_up_to_next_record_idx(full_output_directory):
    record_idx = 0
    while True:
        record_filename = index_to_record_name(record_idx)
        full_filename = full_output_directory / record_filename
        if not full_filename.exists():
            break
        record_idx += 1
    return record_idx
