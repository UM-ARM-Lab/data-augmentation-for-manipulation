#!/usr/bin/env python
import argparse
import logging
from time import perf_counter
import pathlib

import colorama
import numpy as np
import tensorflow as tf

from arc_utilities import ros_init
from link_bot_data.balance import balance
from link_bot_data.classifier_dataset import ClassifierDatasetLoader
from link_bot_data.dataset_utils import batch_tf_dataset


@ros_init.with_ros("debugging_balancing")
def main():
    colorama.init(autoreset=True)
    tf.get_logger().setLevel(logging.ERROR)

    np.set_printoptions(linewidth=250, precision=4, suppress=True)
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')

    args = parser.parse_args()

    dataset = ClassifierDatasetLoader(dataset_dirs=args.dataset_dirs,
                                      load_true_states=True,
                                      use_gt_rope=True,
                                      threshold=0.07,
                                      old_compat=False)
    tf_dataset = dataset.get_datasets(mode='train')
    batch_size = 24
    n = 100
    tf_dataset = batch_tf_dataset(tf_dataset, batch_size, drop_remainder=True)

    n_pos = 0
    n_neg = 0
    t0 = perf_counter()
    for example in tf_dataset.take(n):
        n_pos += tf.reduce_sum(tf.cast(example['is_close'][:, 1] > 0.5, dtype=tf.int64)).numpy()
        n_neg += tf.reduce_sum(tf.cast(example['is_close'][:, 1] < 0.5, dtype=tf.int64)).numpy()
    print('not balanced', n_pos, n_neg, perf_counter() - t0)

    tf_dataset = dataset.get_datasets(mode='train')
    tf_dataset = tf_dataset.balance()
    tf_dataset = batch_tf_dataset(tf_dataset, batch_size, drop_remainder=True)

    n_pos = 0
    n_neg = 0
    t0 = perf_counter()
    for example in tf_dataset.take(n):
        n_pos += tf.reduce_sum(tf.cast(example['is_close'][:, 1] > 0.5, dtype=tf.int64)).numpy()
        n_neg += tf.reduce_sum(tf.cast(example['is_close'][:, 1] < 0.5, dtype=tf.int64)).numpy()
    print('balanced', n_pos, n_neg, perf_counter() - t0)
    # GRRRR WHY IS THIS SO SLOW


if __name__ == '__main__':
    main()
