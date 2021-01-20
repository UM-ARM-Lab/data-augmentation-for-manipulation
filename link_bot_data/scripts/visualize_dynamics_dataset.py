#!/usr/bin/env python
import argparse
import pathlib
from collections import OrderedDict

import colorama
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import rospy
from link_bot_data.dynamics_dataset import DynamicsDatasetLoader
from link_bot_pycommon.args import my_formatter
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.indexing import index_time_np
from moonshine.moonshine_utils import numpify
from sensor_msgs.msg import Image


def plot_3d(args, dataset: DynamicsDatasetLoader, tf_dataset: tf.data.Dataset):
    scenario = dataset.scenario
    image_diff_viz_pub = rospy.Publisher("image_diff_viz", Image, queue_size=10, latch=True)
    for i, example in enumerate(tf_dataset):
        if args.start_at is not None and i < args.start_at:
            continue

        example = numpify(example)

        print(f"Example {i}, Trajectory #{int(example['traj_idx'])}")

        time_steps = example['time_idx']
        anim = RvizAnimationController(time_steps)

        while not anim.done:
            t = anim.t()
            scenario.plot_environment_rviz(example)
            example_t = index_time_np(example, dataset.time_indexed_keys, t)
            scenario.plot_state_rviz(example_t, label='')
            scenario.plot_action_rviz_internal(example_t, label='')

            if t < dataset.steps_per_traj - 1:
                s_next = index_time_np(example, dataset.time_indexed_keys, t + 1)
                # diff = s['rgbd'][:, :, :3] - s_next['rgbd'][:, :, :3]
                # publish_color_image(image_diff_viz_pub, diff)

            # this will return when either the animation is "playing" or because the user stepped forward
            anim.step()


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

    rospy.init_node("visualize_dynamics_dataset")

    np.random.seed(1)
    tf.random.set_seed(1)

    # load the dataset
    dataset = DynamicsDatasetLoader(args.dataset_dir)
    tf_dataset = dataset.get_datasets(mode=args.mode, take=args.take)

    if args.shuffle:
        tf_dataset = tf_dataset.shuffle(1024, seed=1)

    # print info about shapes
    example = next(iter(tf_dataset))
    print("Example:")
    for k, v in example.items():
        if hasattr(v, 'shape'):
            print(k, v.shape)
        elif isinstance(v, OrderedDict):
            print(k, numpify(v))
        else:
            print(k, v)

    if args.plot_type == '3d':
        # uses rviz
        plot_3d(args, dataset, tf_dataset)
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
