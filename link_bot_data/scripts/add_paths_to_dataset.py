#!/usr/bin/env python
import argparse
import pathlib

import colorama
import tensorflow as tf
from colorama import Fore
from progressbar import progressbar

import rospy
from link_bot_data.classifier_dataset import ClassifierDatasetLoader
from link_bot_data.dataset_utils import tf_write_features, float_tensor_to_bytes_feature, bytes_feature
from link_bot_data.dynamics_dataset import DynamicsDatasetLoader
from link_bot_data.modify_dataset import modify_hparams
from link_bot_data.progressbar_widgets import mywidgets
from link_bot_data.recovery_dataset import RecoveryDatasetLoader
from link_bot_pycommon.args import my_formatter


def main():
    colorama.init(autoreset=True)

    parser = argparse.ArgumentParser(description="adds file path to the example", formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('dataset_type', choices=['dy', 'cl', 'rcv'], help='dataset type')

    args = parser.parse_args()

    rospy.init_node("add_paths")

    outdir = args.dataset_dir.parent / f"{args.dataset_dir.name}+paths"

    if args.dataset_type == 'dy':
        dataset = DynamicsDatasetLoader([args.dataset_dir])
    elif args.dataset_type == 'cl':
        dataset = ClassifierDatasetLoader([args.dataset_dir], load_true_states=True, use_gt_rope=False)
    elif args.dataset_type == 'rcv':
        dataset = RecoveryDatasetLoader([args.dataset_dir])
    else:
        raise NotImplementedError(f"Invalid dataset type {args.dataset_type}")

    # hparams
    hparams_update = {'has_tfrecord_path': True}
    modify_hparams(args.dataset_dir, outdir, hparams_update)

    total_count = 0
    for mode in ['train', 'test', 'val']:
        tf_dataset = dataset.get_datasets(mode=mode, shuffle_files=False, do_not_process=True)
        full_output_directory = outdir / mode
        full_output_directory.mkdir(parents=True, exist_ok=True)

        for example, tfrecord_path in zip(progressbar(tf_dataset, widgets=mywidgets), tf_dataset.records):
            features = {k: float_tensor_to_bytes_feature(v) for k, v in example.items()}
            features['tfrecord_path'] = bytes_feature(
                tf.io.serialize_tensor(tf.convert_to_tensor(tfrecord_path, dtype=tf.string)).numpy())
            tf_write_features(full_output_directory, features, total_count)
            total_count += 1
    print(Fore.GREEN + f"Modified {total_count} examples")


if __name__ == '__main__':
    main()
