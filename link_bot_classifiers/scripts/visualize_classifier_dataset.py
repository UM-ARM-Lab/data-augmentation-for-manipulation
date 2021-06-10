#!/usr/bin/env python
import argparse
import pathlib

import colorama
import matplotlib.pyplot as plt
import numpy as np

from arc_utilities import ros_init
from link_bot_classifiers.visualize_classifier_dataset import visualize_dataset
from link_bot_data.load_dataset import load_classifier_dataset
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(1)


@ros_init.with_ros("visualize_classifier_dataset")
def main():
    colorama.init(autoreset=True)

    plt.style.use("slides")
    np.set_printoptions(suppress=True, linewidth=200, precision=3)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    parser.add_argument('--display-type', choices=['just_count', '3d', 'stdev'], default='3d')
    parser.add_argument('--mode', choices=['train', 'val', 'test', 'all'], default='train')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--start-at', type=int, default=0)
    parser.add_argument('--take', type=int)
    parser.add_argument('--only-negative', action='store_true')
    parser.add_argument('--only-positive', action='store_true')
    parser.add_argument('--only-infeasible', action='store_true')
    parser.add_argument('--only-in-collision', action='store_true')
    parser.add_argument('--only-starts-far', action='store_true')
    parser.add_argument('--only-reconverging', action='store_true')
    parser.add_argument('--perf', action='store_true', help='print time per iteration')
    parser.add_argument('--no-plot', action='store_true', help='only print statistics')

    args = parser.parse_args()
    args.batch_size = 1

    classifier_dataset = load_classifier_dataset(args.dataset_dirs, load_true_states=True, threshold=args.threshold)

    visualize_dataset(args, classifier_dataset)


if __name__ == '__main__':
    main()
