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
from link_bot_data.coerce_types import coerce_types
from link_bot_data.tf_dataset_utils import ros_msg_to_bytes_feature, generic_to_bytes_feature, parse_dataset, \
    slow_deserialize, is_reconverging, num_reconverging, \
    tf_write_example
from link_bot_pycommon import pycommon
from link_bot_pycommon.grid_utils import pad_voxel_grid
from link_bot_pycommon.serialization import dump_gzipped_pickle
from moonshine.filepath_tools import load_params
from moonshine.moonshine_utils import remove_batch, add_batch
from moonshine.numpify import numpify

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


def add_predicted_cond(feature_name: str, cond):
    return PREDICTED_PREFIX + feature_name if cond else feature_name


def add_new(feature_name: str):
    return NEW_PREFIX + feature_name


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


def index_to_filename2(traj_idx, save_format):
    if save_format == 'pkl':
        return index_to_filename('.pkl', traj_idx)
    elif save_format == 'tfrecord':
        return index_to_record_name(traj_idx)


def train_test_split_counts(n: int, val_split: int = DEFAULT_VAL_SPLIT, test_split: int = DEFAULT_TEST_SPLIT):
    n_test = int(test_split * n)
    n_val = int(val_split * n)
    n_train = n - n_test - n_val
    return n_train, n_val, n_test


def compute_batch_size_for_n_examples(total_examples: int, max_batch_size: int):
    batch_size = min(max(1, int(total_examples / 2)), max_batch_size)
    return batch_size


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
        return tf_write_example(full_output_directory, example, example_idx)
    elif save_format == 'pkl':
        return pkl_write_example(full_output_directory, example, example_idx, extra_metadata_keys)
    else:
        raise NotImplementedError()
