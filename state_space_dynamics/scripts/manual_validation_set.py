#!/usr/bin/env python
import argparse
import pathlib

import numpy as np

from arc_utilities import ros_init
from link_bot_data.tf_dataset_utils import deserialize_scene_msg, index_to_filename
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
    parser.add_argument('--shuffle', action='store_true', help='shuffle')
    parser.add_argument('--weight-above', type=float, default=0)
    parser.add_argument('--weight-below', type=float, default=1)

    args = parser.parse_args()

    # load the dataset
    dataset = TorchDynamicsDataset(args.dataset_dir, mode='notrain')
    indices = np.arange(0, len(dataset))

    s = dataset.get_scenario()

    dataset_anim = RvizAnimationController(time_steps=indices, ns='trajs')

    indices_to_keep = []

    while not dataset_anim.done:
        example_idx = dataset_anim.t()
        example = dataset[example_idx]
        if 'traj_idx' in example:
            traj_idx = example['traj_idx']
            s.plot_traj_idx_rviz(traj_idx)

        example = numpify(example)

        print(indices_to_keep)

        dataset.anim_rviz(example)
        q = input("Press [r] to remove, [k] to keep, [p] to show it again, and anything else to skip? ")
        if q == 'r':
            indices_to_keep.remove(example_idx)
        elif q == 'k':
            indices_to_keep.append(example_idx)
        elif q == 'p':
            continue

        dataset_anim.step()

    for i in indices_to_keep:
        print(index_to_filename('.pkl', i))


if __name__ == '__main__':
    main()
