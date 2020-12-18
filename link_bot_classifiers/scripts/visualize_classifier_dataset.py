#!/usr/bin/env python
import argparse
import pathlib
from time import perf_counter

import colorama
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from progressbar import progressbar
from scipy import stats

import rospy
from link_bot_data import base_dataset
from link_bot_data.classifier_dataset import ClassifierDatasetLoader
from link_bot_data.dataset_utils import add_predicted
from link_bot_pycommon.pycommon import print_dict
from moonshine.gpu_config import limit_gpu_mem
from moonshine.moonshine_utils import remove_batch

limit_gpu_mem(1)


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
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--start-at', type=int, default=0)
    parser.add_argument('--take', type=int)
    parser.add_argument('--use-gt-rope', action='store_true')
    parser.add_argument('--old-compat', action='store_true')
    parser.add_argument('--only-negative', action='store_true')
    parser.add_argument('--only-positive', action='store_true')
    parser.add_argument('--only-in-collision', action='store_true')
    parser.add_argument('--only-reconverging', action='store_true')
    parser.add_argument('--perf', action='store_true', help='print time per iteration')
    parser.add_argument('--no-plot', action='store_true', help='only print statistics')

    args = parser.parse_args()
    args.batch_size = 1

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    rospy.init_node("visualize_classifier_data")

    classifier_dataset = ClassifierDatasetLoader(args.dataset_dirs,
                                                 load_true_states=True,
                                                 threshold=args.threshold,
                                                 use_gt_rope=args.use_gt_rope,
                                                 old_compat=args.old_compat)

    visualize_dataset(args, classifier_dataset)


def visualize_dataset(args, classifier_dataset):
    tf_dataset = classifier_dataset.get_datasets(mode=args.mode, take=args.take)

    tf_dataset = tf_dataset.batch(1)

    iterator = iter(tf_dataset)
    t0 = perf_counter()

    reconverging_count = 0
    positive_count = 0
    negative_count = 0
    count = 0

    stdevs = []
    labels = []
    stdevs_for_negative = []
    stdevs_for_positive = []

    for i, example in enumerate(tf_dataset):
    # for i, example in enumerate(progressbar(tf_dataset, widgets=base_dataset.widgets)):
        if i < args.start_at:
            continue

        example = remove_batch(example)

        is_close = example['is_close'].numpy().squeeze()
        count += is_close.shape[0]

        n_close = np.count_nonzero(is_close[-1])
        n_far = is_close.shape[0] - n_close
        positive_count += n_close
        negative_count += n_far
        reconverging = n_far > 0 and is_close[-1]

        if args.only_reconverging and not reconverging:
            continue

        if args.only_negative and np.any(is_close[1:]):
            continue

        if args.only_positive and not np.any(is_close[1:]):
            continue

        # print(f"Example {i}, Trajectory #{int(example['traj_idx'])}")

        if count == 0:
            print_dict(example)

        if reconverging:
            reconverging_count += 1

        print(i, is_close[1])

        # Print statistics intermittently
        # if count % 1000 == 0:
        #     print_stats_and_timing(args, count, reconverging_count, negative_count, positive_count)

        #############################
        # Show Visualization
        #############################
        if args.display_type == 'just_count':
            continue
        elif args.display_type == '3d':
            # print(example['is_close'])
            if example['is_close'][0] == 0:
                continue
            classifier_dataset.anim_transition_rviz(example)

        elif args.display_type == 'stdev':
            for t in range(1, classifier_dataset.horizon):
                stdev_t = example[add_predicted('stdev')][t, 0].numpy()
                label_t = example['is_close'][t]
                stdevs.append(stdev_t)
                labels.append(label_t)
                if label_t > 0.5:
                    stdevs_for_positive.append(stdev_t)
                else:
                    stdevs_for_negative.append(stdev_t)
        else:
            raise NotImplementedError()
    total_dt = perf_counter() - t0

    if args.display_type == 'stdev':
        print(f"p={stats.f_oneway(stdevs_for_negative, stdevs_for_positive)[1]}")

        plt.figure()
        plt.title(" ".join([str(d.name) for d in args.dataset_dirs]))
        bins = plt.hist(stdevs_for_negative, label='negative examples', alpha=0.8, density=True)[1]
        plt.hist(stdevs_for_positive, label='positive examples', alpha=0.8, bins=bins, density=True)
        plt.ylabel("count")
        plt.xlabel("stdev")
        plt.legend()
        plt.show()

    print_stats_and_timing(args, count, reconverging_count, negative_count, positive_count, total_dt)


def print_stats_and_timing(args, count, reconverging_count, negative_count, positive_count, total_dt=None):
    if args.perf and total_dt is not None:
        print("Total iteration time = {:.4f}".format(total_dt))
    class_balance = positive_count / count * 100
    print("Number of examples: {}".format(count))
    print("Number of reconverging examples: {}".format(reconverging_count))
    print("Number positive: {}".format(positive_count))
    print("Number negative: {}".format(negative_count))
    print("Class balance: {:4.1f}% positive".format(class_balance))


if __name__ == '__main__':
    main()
