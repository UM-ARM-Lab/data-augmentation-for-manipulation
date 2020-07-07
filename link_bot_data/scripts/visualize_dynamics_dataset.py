#!/usr/bin/env python

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import rospy
from link_bot_data.dynamics_dataset import DynamicsDataset
from link_bot_pycommon.animation_player import Player
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.rviz_animation_controller import RvizAnimationController
from moonshine.moonshine_utils import numpify, add_batch, remove_batch


def plot_3d(dataset: DynamicsDataset, tf_dataset: tf.data.Dataset):
    rospy.loginfo("Don't forget to start the viz_stepper")
    min_x = 1000
    min_y = 1000
    min_z = 1000
    max_x = 0
    max_y = 0
    max_z = 0
    for i, example in enumerate(tf_dataset):
        example = numpify(example)
        time_steps = example['time_idx']

        if i < 95:
            continue
        # for t in time_steps:
        #     example_t = remove_batch(dataset.index_time(add_batch(example), t))
        #     x, y, z = example_t['gripper1']
        #     max_x = max(max_x, x)
        #     max_y = max(max_y, y)
        #     max_z = max(max_z, z)
        #     min_x = min(min_x, x)
        #     min_y = min(min_y, y)
        #     min_z = min(min_z, z)
        #     x, y, z = example_t['gripper2']
        #     min_x = min(min_x, x)
        #     min_y = min(min_y, y)
        #     min_z = min(min_z, z)
        #     max_x = max(max_x, x)
        #     max_y = max(max_y, y)
        #     max_z = max(max_z, z)

        dataset.scenario.plot_environment_rviz(example)
        anim = RvizAnimationController(time_steps)
        while not anim.done:
            t = anim.t()
            example_t = remove_batch(dataset.index_time(add_batch(example), t))
            dataset.scenario.plot_state_rviz(example_t, label='')
            dataset.scenario.plot_action_rviz_internal(example_t, label='')

            # this will return when either the animation is "playing" or because the user stepped forward
            anim.step()

    #     if i % 100 == 0:
    #         print(min_x, max_x)
    #         print(min_y, max_y)
    #         print(min_z, max_z)
    # print(min_x, max_x)
    # print(min_y, max_y)
    # print(min_z, max_z)


def main():
    plt.style.use("slides")
    np.set_printoptions(suppress=True, linewidth=250, precision=5)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory', nargs='+')
    parser.add_argument('plot_type', choices=['2d', '3d', 'sanity_check', 'just_count'], default='2d')
    parser.add_argument('--take', type=int)
    parser.add_argument('--mode', choices=['train', 'test', 'val', 'all'], default='train', help='train test or val')
    parser.add_argument('--shuffle', action='store_true', help='shuffle')

    args = parser.parse_args()

    rospy.init_node("visualize_dynamics_dataset")

    np.random.seed(1)
    tf.random.set_seed(1)

    # load the dataset
    dataset = DynamicsDataset(args.dataset_dir)
    tf_dataset = dataset.get_datasets(mode=args.mode, take=args.take)

    if args.shuffle:
        tf_dataset = tf_dataset.shuffle(1024, seed=1)

    # print info about shapes
    example = next(iter(tf_dataset))
    print("Example:")
    for k, v in example.items():
        print(k, v.shape)

    if args.plot_type == '2d':
        plot_2d(dataset, tf_dataset)
    elif args.plot_type == '3d':
        # uses rviz
        plot_3d(dataset, tf_dataset)
    elif args.plot_type == 'sanity_check':
        min_x = 100
        max_x = -100
        min_y = 100
        max_y = -100
        min_z = 100
        max_z = -100
        min_d = 100
        max_d = -100
        for example in tf_dataset:
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
        i = 0
        for _ in tf_dataset:
            i += 1
        print(f'num examples {i}')


if __name__ == '__main__':
    main()
