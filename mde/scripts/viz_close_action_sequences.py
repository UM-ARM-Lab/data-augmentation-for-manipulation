#!/usr/bin/env python
import argparse
import pathlib

import numpy as np
import torch
from tqdm import tqdm

from arc_utilities import ros_init
from link_bot_data.new_dataset_utils import fetch_udnn_dataset
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.torch_geometry import pairwise_squared_distances
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

    ref_actions_list = get_actions_list(ref_dataset)
    train_actions_list = get_actions_list(train_dataset)

    viz_data = compute_viz_data(ref_actions_list, train_actions_list)

    vizualize_data(viz_data, s)


def vizualize_data(viz_data, s):
    anim = RvizAnimationController(n_time_steps=len(viz_data))
    while not anim.done:
        t = anim.t()
        ref_before, ref_after, train_before, train_after, d = viz_data[t]
        s.plot_error_rviz(d)
        s.plot_points_rviz(ref_before, label='ref', color='white', scale=0.01)
        ref_arrow_dir = ref_after - ref_before
        s.plot_arrows_rviz(ref_before, ref_arrow_dir, label='ref', color='white', scale=2)
        s.plot_points_rviz(train_before, label='train', color='blue', scale=0.01)
        train_arrow_dir = train_after - train_before
        s.plot_arrows_rviz(train_before, train_arrow_dir, label='train', color='blue', scale=2)
        anim.step()


def compute_viz_data(ref_actions_list, train_actions_list):
    ref_actions = torch.tensor(ref_actions_list).cuda()
    train_actions = torch.tensor(train_actions_list).cuda()
    train_actions_batched = train_actions.permute([1, 2, 0, 3])
    ref_actions_batched = ref_actions.permute([1, 2, 0, 3])
    distances_to_ref_matrix_all = pairwise_squared_distances(train_actions_batched, ref_actions_batched).sqrt()
    _, _, a, b = distances_to_ref_matrix_all.shape
    distances_to_ref_matrix_flat = distances_to_ref_matrix_all.reshape([4, a, b])
    distances_to_ref_matrix = distances_to_ref_matrix_flat[1:].mean(0)
    # we want to compute the distance between each left/right before/after separately, so treat them as batch dims?
    min_distances, min_indices = distances_to_ref_matrix.min(1)
    data = []
    for train_index, (min_distance, min_ref_index) in enumerate(zip(min_distances, min_indices)):
        train_action = train_actions[train_index]
        train_before = train_action[0]
        train_after = train_action[1]

        ref_action = ref_actions[min_ref_index]
        ref_before = ref_action[0]
        ref_after = ref_action[1]

        if min_distance < 0.04:
            data.append([ref_before, ref_after, train_before, train_after, min_distance])

    print(f"{len(data) / len(train_actions_list):%}")
    return data


def get_actions_list(dataset):
    actions_list = []
    for traj in tqdm(dataset):
        s_0 = dataset.index_time(traj, 0)
        left_gripper_0 = s_0['left_gripper']
        right_gripper_0 = s_0['right_gripper']
        before = np.stack([left_gripper_0, right_gripper_0])

        traj_len = len(traj['time_idx'])
        for t in range(traj_len):
            s_t = dataset.index_time(traj, t)
            left_gripper_t = s_t['left_gripper_position']
            right_gripper_t = s_t['right_gripper_position']
            after = np.stack([left_gripper_t, right_gripper_t])
            origin = before[0]
            before_local = before - origin
            after_local = after - origin
            actions = np.stack([before_local, after_local])
            actions_list.append(actions)

            before = after
    return actions_list


if __name__ == '__main__':
    main()
