#!/usr/bin/env python
import argparse
import pathlib
import pickle

import numpy as np
from tqdm import tqdm

from arc_utilities import ros_init
from link_bot_data.new_dataset_utils import fetch_udnn_dataset
from state_space_dynamics.torch_dynamics_dataset import TorchDynamicsDataset


@ros_init.with_ros("vis_close_action_sequences")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ref_dataset', type=pathlib.Path)
    parser.add_argument('train_dataset', type=pathlib.Path)

    args = parser.parse_args()

    train_dataset = TorchDynamicsDataset(fetch_udnn_dataset(args.train_dataset), mode='notest')
    ref_dataset = TorchDynamicsDataset(fetch_udnn_dataset(args.ref_dataset), mode='test')
    s = train_dataset.get_scenario({'rope_name': 'rope_3d_alt'})

    # the idea is to get a sense of which transitions in the training data are "relevant" for doing well on the
    # reference trajectories (known_good_4). We want to find the closest transition in the training set
    # for each transition in the reference set, and visualize it
    closest_data = []
    for ref_traj_i, ref_traj in enumerate(tqdm(ref_dataset)):
        if ref_traj_i < 70:
            continue

        ref_s_0 = ref_dataset.index_time(ref_traj, 0)
        ref_left_gripper_0 = ref_s_0['left_gripper']
        ref_right_gripper_0 = ref_s_0['right_gripper']
        ref_before = np.stack([ref_left_gripper_0, ref_right_gripper_0])

        ref_traj_len = len(ref_traj['time_idx'])
        for ref_t in range(ref_traj_len):
            ref_s_t = train_dataset.index_time(ref_traj, ref_t)
            ref_left_gripper_t = ref_s_t['left_gripper_position']
            ref_right_gripper_t = ref_s_t['right_gripper_position']
            ref_after = np.stack([ref_left_gripper_t, ref_right_gripper_t])
            ref_points = np.concatenate([ref_before, ref_after])

            s.plot_points_rviz(ref_before, label='ref', color='white', scale=0.01)
            ref_arrow_dir = ref_after - ref_before
            s.plot_arrows_rviz(ref_points, ref_arrow_dir, label='ref', color='white', scale=2)

            closest_distance = np.inf
            train_points = None
            for train_traj in tqdm(train_dataset, position=2):
                train_s_0 = train_dataset.index_time(train_traj, 0)
                train_left_gripper_0 = train_s_0['left_gripper']
                train_right_gripper_0 = train_s_0['right_gripper']
                train_before = np.stack([train_left_gripper_0, train_right_gripper_0])

                train_traj_len = len(train_traj['time_idx'])
                for train_t in range(train_traj_len):
                    train_s_t = train_dataset.index_time(train_traj, train_t)
                    train_left_gripper_t = train_s_t['left_gripper_position']
                    train_right_gripper_t = train_s_t['right_gripper_position']
                    train_after = np.stack([train_left_gripper_t, train_right_gripper_t])

                    train_points = np.concatenate([train_before, train_after])

                    d = np.linalg.norm(train_points - ref_points, axis=-1).mean()
                    if d < closest_distance:
                        s.plot_points_rviz(train_before, label='train', color='blue', scale=0.01)
                        train_arrow_dir = train_after - train_before
                        s.plot_arrows_rviz(train_points, train_arrow_dir, label='train', color='blue', scale=2)
                        closest_distance = d

                    train_before = train_after

            print(f"{closest_distance=}")
            closest_data.append({
                'ref':   ref_points,
                'train': train_points,
            })

            ref_before = ref_after

    with open(f"results/closest_action_sequences_{args.ref_dataset}_{args.train_dataset}.pkl", 'wb') as f:
        pickle.dump(closest_data, f)


if __name__ == '__main__':
    main()
