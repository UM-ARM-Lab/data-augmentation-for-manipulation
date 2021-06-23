#!/usr/bin/env python
import argparse
import logging
import pathlib

import tensorflow as tf

from arc_utilities import ros_init
from link_bot_classifiers.iterative_fine_tuning import iterative_fine_tuning


@ros_init.with_ros("fine_tune_classifier")
def main():
    tf.get_logger().setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument('training_dataset_dir', type=pathlib.Path)
    parser.add_argument('checkpoint', type=pathlib.Path)
    parser.add_argument('proxy_datasets_info', type=pathlib.Path, help='a hjson file listing dataset and metric names')
    parser.add_argument('nickname')
    parser.add_argument('--params', '-p', type=pathlib.Path, help='an hjson file to override the model hparams')
    parser.add_argument('--augmentation-config-dir', type=pathlib.Path, help='dir of pkl files with state/env')
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--epochs', type=int, default=10)

    args = parser.parse_args()

    iterative_fine_tuning(**vars(args))


if __name__ == '__main__':
    main()
