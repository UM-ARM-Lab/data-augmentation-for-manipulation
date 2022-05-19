#!/usr/bin/env python
import argparse
import pathlib

import numpy as np

import rospy
from arc_utilities import ros_init
from link_bot_data.tf_dataset_utils import deserialize_scene_msg
from link_bot_pycommon.args import int_set_arg
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.numpify import numpify
from state_space_dynamics.torch_dynamics_dataset import TorchDynamicsDataset


@ros_init.with_ros("visualize_dynamics_dataset")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('--take', type=int)
    parser.add_argument('--skip', type=int)
    parser.add_argument('--shard', type=int)
    parser.add_argument('--mode', default='all', help='train test or val')
    parser.add_argument('--shuffle', action='store_true', help='shuffle')
    parser.add_argument('--indices', type=int_set_arg, help='show specific example by example_idx value')

    args = parser.parse_args()

    # load the dataset
    dataset = TorchDynamicsDataset(args.dataset_dir, mode=args.mode)
    indices = np.arange(0, len(dataset))
    if args.shuffle:
        np.random.shuffle(indices)

    if args.take:
        indices = indices[:args.take]

    if args.skip:
        indices = indices[args.skip:]

    if args.shard:
        indices = indices[::args.shard]

    # print info about shapes
    dataset.pprint_example()

    s = dataset.get_scenario()

    dataset_anim = RvizAnimationController(time_steps=indices, ns='trajs')

    n_examples_visualized = 0
    while not dataset_anim.done:
        example_idx = dataset_anim.t()
        example = dataset[example_idx]
        if args.indices is not None:
            rospy.logwarn_once(f"Got indices: {args.indices}")
            if example['example_idx'] not in args.indices:
                dataset_anim.step()
                continue
        print(example['example_idx'])
        if 'traj_idx' in example:
            traj_idx = example['traj_idx']
            s.plot_traj_idx_rviz(traj_idx)
        if 'meta_mask' in example:
            print(example['meta_mask'])

        deserialize_scene_msg(example)
        if 'augmented_from' in example:
            print(f"augmented from: {example['augmented_from']}")
        example = numpify(example)
        dataset.anim_rviz(example)

        n_examples_visualized += 1
        dataset_anim.step()

    print(f"{n_examples_visualized:=}")


if __name__ == '__main__':
    main()
