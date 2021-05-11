#!/usr/bin/env python
import argparse
import pathlib

import colorama
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from progressbar import progressbar

import rospy
from link_bot_data import base_dataset
from link_bot_data.dataset_utils import pprint_example
from link_bot_data.recovery_dataset import RecoveryDatasetLoader
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(1)


def main():
    colorama.init(autoreset=True)

    plt.style.use("slides")
    np.set_printoptions(suppress=True, linewidth=200, precision=5)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    parser.add_argument('--type', choices=['best_to_worst', 'in_order', 'stats'], default='best_to_worst')
    parser.add_argument('--mode', choices=['train', 'val', 'test', 'all'], default='train')

    args = parser.parse_args()

    rospy.init_node('vis_recovery_dataset')

    dataset = RecoveryDatasetLoader(args.dataset_dirs)
    if args.type == 'stats':
        stats(args, dataset)
    else:
        if args.type == 'best_to_worst':
            tf_dataset = dataset.get_datasets(mode=args.mode, sort=True)
        else:
            tf_dataset = dataset.get_datasets(mode=args.mode)

        example = next(iter(tf_dataset))
        print("Example:")
        pprint_example(example)

        for example in progressbar(tf_dataset, widgets=base_dataset.widgets):
            n_accepts = tf.math.count_nonzero(example['accept_probabilities'] > 0.5, axis=1)
            # if not is_stuck(example):
            #     print("found a not-stuck example")
            #     dataset.anim_rviz(example)
            dataset.anim_rviz(example)


def stats(args, dataset):
    recovery_probabilities = []
    batch_size = 512
    tf_dataset = dataset.get_datasets(mode=args.mode).batch(batch_size, drop_remainder=True)
    for example in tf_dataset:
        recovery_probabilities.append(tf.reduce_mean(example['recovery_probability'][:, 1]))

    overall_recovery_probability_mean = tf.reduce_mean(recovery_probabilities)

    print(f'mean recovery probability of dataset: {overall_recovery_probability_mean:.5f}')

    losses = []
    for example in tf_dataset:
        y_true = tf.reshape(example['recovery_probability'][:, 1], [batch_size, 1])
        pred = tf.reshape([overall_recovery_probability_mean] * batch_size, [batch_size, 1])
        loss = tf.keras.losses.binary_crossentropy(y_true=y_true, y_pred=pred, from_logits=False)
        losses.append(loss)
    print(f"loss to beat {tf.reduce_mean(losses)}")


if __name__ == '__main__':
    main()
