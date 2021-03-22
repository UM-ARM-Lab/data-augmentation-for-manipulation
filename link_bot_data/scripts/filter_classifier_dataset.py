#!/usr/bin/env python
import argparse
import inspect
import pathlib
from typing import Dict

import colorama
import tensorflow as tf

import rospy
from link_bot_data.classifier_dataset import ClassifierDatasetLoader
from link_bot_data.modify_dataset import filter_dataset
from link_bot_pycommon.args import my_formatter


def starts_close(dataset: ClassifierDatasetLoader, example: Dict):
    starts_close = (example['is_close'][0] == 1)
    return starts_close


def starts_far(dataset: ClassifierDatasetLoader, example: Dict):
    starts_far = (example['is_far'][0] == 0)
    return starts_far


def is_feasible(dataset: ClassifierDatasetLoader, example: Dict):
    joint_pos_dist = tf.linalg.norm(example['joint_positions'] - example['predicted/joint_positions'])
    feasible = joint_pos_dist < 0.075
    return feasible


filter_funcs = {name: f for name, f in vars().items() if inspect.isfunction(f)}


def main():
    colorama.init(autoreset=True)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('filter_func_name', type=str, help='name of one of the above free functions for filtering')
    parser.add_argument('suffix', type=str, help='string added to the new dataset name')

    args = parser.parse_args()

    rospy.init_node("modify_dynamics_dataset")

    outdir = args.dataset_dir.parent / f"{args.dataset_dir.name}+{args.suffix}"

    hparams_update = {}

    if args.filter_func_name in filter_funcs:
        should_keep = filter_funcs[args.filter_func_name]
    else:
        print(f"No available function {args.filter_func_name}")
        print(f"Available functions are:")
        print(filter_funcs)
        return

    dataset = ClassifierDatasetLoader([args.dataset_dir], use_gt_rope=False, load_true_states=True)
    filter_dataset(dataset_dir=args.dataset_dir,
                   dataset=dataset,
                   outdir=outdir,
                   should_keep=should_keep,
                   hparams_update=hparams_update,
                   do_not_process=False)


if __name__ == '__main__':
    main()
