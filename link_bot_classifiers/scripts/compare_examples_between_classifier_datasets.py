#!/usr/bin/env python
import argparse
import pathlib

import colorama
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import rospy
from link_bot_classifiers.visualize_classifier_dataset import compare_examples_from_datasets
from link_bot_data.classifier_dataset import ClassifierDatasetLoader
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(1)


def main():
    colorama.init(autoreset=True)

    plt.style.use("slides")
    np.set_printoptions(suppress=True, linewidth=200, precision=3)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir1', type=pathlib.Path)
    parser.add_argument('dataset_dir2', type=pathlib.Path)
    parser.add_argument('--example-indices', type=int, nargs='+', help='only consider these specific indices')
    parser.add_argument('--show-all', action='store_true')
    parser.add_argument('--old-compat', action='store_true')
    parser.add_argument('--mode', choices=['train', 'val', 'test', 'all'], default='train')
    parser.add_argument('--take', type=int)

    args = parser.parse_args()
    args.batch_size = 1

    rospy.init_node("compare_examples_between_classifier_datasets")

    classifier_dataset1 = ClassifierDatasetLoader([args.dataset_dir1],
                                                  load_true_states=True,
                                                  old_compat=args.old_compat)

    classifier_dataset2 = ClassifierDatasetLoader([args.dataset_dir2],
                                                  load_true_states=True,
                                                  old_compat=args.old_compat)

    compare_examples_from_datasets(args, classifier_dataset1, classifier_dataset2)


if __name__ == '__main__':
    main()
