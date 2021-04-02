#!/usr/bin/env python
import argparse
import pathlib

import colorama
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from progressbar import progressbar

from arc_utilities import ros_init
from link_bot_data import base_dataset
from link_bot_data.dataset_utils import deserialize_scene_msg, pprint_example
from link_bot_data.dynamics_dataset import DynamicsDatasetLoader
from link_bot_pycommon.args import my_formatter
from moonshine.moonshine_utils import numpify


@ros_init.with_ros("visualize_dynamics_dataset")
def main():
    colorama.init(autoreset=True)
    plt.style.use("slides")
    np.set_printoptions(suppress=True, linewidth=250, precision=5)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory', nargs='+')
    parser.add_argument('--plot-type', choices=['3d', 'sanity_check', 'just_count'], default='3d')
    parser.add_argument('--take', type=int)
    parser.add_argument('--start-at', type=int)
    parser.add_argument('--mode', choices=['train', 'test', 'val', 'all'], default='all', help='train test or val')
    parser.add_argument('--shuffle', action='store_true', help='shuffle')

    args = parser.parse_args()

    # load the dataset
    dynamics_dataset = DynamicsDatasetLoader(args.dataset_dir)
    tf_dataset = dynamics_dataset.get_datasets(mode=args.mode, take=args.take)

    if args.shuffle:
        tf_dataset = tf_dataset.shuffle(1024, seed=1)

    # print info about shapes
    example = next(iter(tf_dataset))
    print("Example:")
    pprint_example(example)

    for i, example in enumerate(progressbar(tf_dataset, widgets=base_dataset.widgets)):
        if args.start_at is not None and i < args.start_at:
            continue

        traj_idx = example['traj_idx']
        dynamics_dataset.scenario.plot_traj_idx_rviz(traj_idx)

        if args.plot_type == '3d':
            deserialize_scene_msg(example)
            example = numpify(example)
            dynamics_dataset.anim_rviz(example)
        elif args.plot_type == 'sanity_check':
            min_x = 100
            max_x = -100
            min_y = 100
            max_y = -100
            min_z = 100
            max_z = -100
            min_d = 100
            max_d = -100
            distances_between_grippers = tf.linalg.norm(example['gripper2'] - example['gripper1'], axis=-1)
            min_d = min(tf.reduce_min(distances_between_grippers).numpy(), min_d)
            max_d = max(tf.reduce_max(distances_between_grippers).numpy(), max_d)
            rope = example['link_bot']
            points = tf.reshape(rope, [rope.shape[0], -1, 3])
            min_x = min(tf.reduce_min(points[:, :, 0]).numpy(), min_x)
            max_x = max(tf.reduce_max(points[:, :, 0]).numpy(), max_x)
            min_y = min(tf.reduce_min(points[:, :, 1]).numpy(), min_y)
            max_y = max(tf.reduce_max(points[:, :, 1]).numpy(), max_y)
            min_z = min(tf.reduce_min(points[:, :, 2]).numpy(), min_z)
            max_z = max(tf.reduce_max(points[:, :, 2]).numpy(), max_z)
            print(min_d, max_d)
            print(min_x, max_x, min_y, max_y, min_z, max_z)
        elif args.plot_type == 'just_count':
            pass


if __name__ == '__main__':
    main()
