#!/usr/bin/env python
import os
import pathlib
import time
from collections import OrderedDict
from typing import Optional, Dict, Sequence

import git
import numpy as np
from colorama import Fore

from arc_utilities.filesystem_utils import mkdir_and_ask
from link_bot_pycommon import pycommon
from moonshine.grid_utils_tf import pad_voxel_grid
from moonshine.filepath_tools import load_params
from moonshine.numpify import numpify
from moveit_msgs.msg import PlanningScene

NULL_PAD_VALUE = -10000

DEFAULT_VAL_SPLIT = 0.125
DEFAULT_TEST_SPLIT = 0.125


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


def add_predicted_hack(feature_name: str):
    return 'predicted/' + feature_name


def add_predicted_cond(feature_name: str, cond):
    return PREDICTED_PREFIX + feature_name if cond else feature_name


def add_new(feature_name: str):
    return NEW_PREFIX + feature_name


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
        elif isinstance(v, PlanningScene):
            print(k, type(v))
        else:
            print(k, v)


def train_test_split_counts(n: int, val_split: int = DEFAULT_VAL_SPLIT, test_split: int = DEFAULT_TEST_SPLIT):
    n_test = int(test_split * n)
    n_val = int(val_split * n)
    n_train = n - n_test - n_val
    return n_train, n_val, n_test


def compute_batch_size_for_n_examples(total_examples: int, max_batch_size: int):
    batch_size = min(max(1, int(total_examples / 2)), max_batch_size)
    return batch_size


def merge_hparams_dicts(dataset_dirs, verbose: int = 0) -> Dict:
    if isinstance(dataset_dirs, pathlib.Path):
        dataset_dirs = [dataset_dirs]
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
