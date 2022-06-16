#!/usr/bin/env python
import argparse
import pathlib
import pickle
from multiprocessing import get_context

import numpy as np
import torch
from tqdm import tqdm

from arc_utilities import ros_init
from link_bot_data.dataset_utils import add_predicted
from link_bot_data.new_dataset_utils import fetch_mde_dataset
from link_bot_pycommon.args import int_set_arg
from link_bot_pycommon.matplotlib_utils import adjust_lightness
from mde.torch_mde_dataset import TorchMDEDataset
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.indexing import try_index_time, index_time
from moonshine.numpify import numpify
from moonshine.torch_geometry import pairwise_squared_distances


@ros_init.with_ros("vis_close_mde_examples")
def main():
    np.set_printoptions(precision=4)

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=pathlib.Path)
    parser.add_argument('example_indices', type=int_set_arg)

    args = parser.parse_args()

    all_dataset = TorchMDEDataset(fetch_mde_dataset(args.dataset), mode='all')
    s = all_dataset.get_scenario({'rope_name': 'rope_3d_alt'})

    train_dataset = TorchMDEDataset(fetch_mde_dataset(args.dataset), mode='train')

    ref_actions_list = []
    for example_idx in args.example_indices:
        val_traj = all_dataset[example_idx]
        print(f"found ref example {val_traj['example_idx']}")
        actions, _ = get_actions(all_dataset.time_indexed_keys, val_traj)
        ref_actions_list.append(actions)

    cache = pathlib.Path("viz_close_mde_examples.pkl")
    if cache.exists():
        with cache.open('rb') as f:
            train_actions_list, train_example_indices = pickle.load(f)
    else:
        train_actions_list, train_example_indices = get_actions_list(train_dataset)
        with cache.open('wb') as f:
            pickle.dump((train_actions_list, train_example_indices), f)

    ref_example_indices = args.example_indices

    ref_actions = torch.tensor(ref_actions_list).cuda()
    train_actions = torch.tensor(train_actions_list).cuda()
    train_actions_batched = train_actions.permute([1, 2, 0, 3])
    ref_actions_batched = ref_actions.permute([1, 2, 0, 3])
    distances_to_ref_matrix_all = pairwise_squared_distances(train_actions_batched, ref_actions_batched).sqrt()
    _, _, a, b = distances_to_ref_matrix_all.shape
    distances_to_ref_matrix_flat = distances_to_ref_matrix_all.reshape([4, a, b])
    distances_to_ref_matrix = distances_to_ref_matrix_flat[1:].mean(0)
    # we want to compute the distance between each left/right before/after separately, so treat them as batch dims?
    nearest_distances, nearest_indices = distances_to_ref_matrix.min(0)

    nearest_train_example_indices = np.array(train_example_indices)[nearest_indices.cpu().numpy()]

    def get_pred_t(example, _t):
        return numpify(index_time(example, all_dataset.time_indexed_keys_predicted, _t, False))

    def get_actual_t(example, _t):
        return numpify(index_time(example, all_dataset.time_indexed_keys, _t, False))

    anim = RvizAnimationController(n_time_steps=len(ref_example_indices))
    while not anim.done:
        t = anim.t()
        ref_example_index = ref_example_indices[t]
        nearest_train_example_index = nearest_train_example_indices[t]
        ref_example = all_dataset[ref_example_index]
        nearest_train_example = all_dataset[nearest_train_example_index]

        ref_pred_0 = get_pred_t(all_dataset[ref_example_index], 0)
        ref_pred_1 = get_pred_t(all_dataset[ref_example_index], 1)
        ref_pred_1.pop(add_predicted("left_gripper"))
        ref_pred_1.pop(add_predicted("right_gripper"))
        nearest_train_0 = get_pred_t(all_dataset[nearest_train_example_index], 0)
        nearest_train_1 = get_pred_t(all_dataset[nearest_train_example_index], 1)
        nearest_train_1.pop(add_predicted("left_gripper"))
        nearest_train_1.pop(add_predicted("right_gripper"))
        ref_actual_0 = get_actual_t(all_dataset[ref_example_index], 0)
        ref_actual_1 = get_actual_t(all_dataset[ref_example_index], 1)

        s.plot_environment_rviz(ref_example)
        print(f"{ref_example_index=}")
        print(f"validation example has error {ref_example['error']}")
        print(f"but nearest training example has error {nearest_train_example['error']}")
        s.plot_state_rviz(ref_pred_0, label='ref_pred_0', color='red')
        s.plot_state_rviz(ref_pred_1, label='ref_pred_1', color=adjust_lightness('red', 0.5))
        s.plot_action_rviz(ref_pred_0, ref_pred_1, label='ref', color='red')
        s.plot_state_rviz(ref_actual_0, label='ref_actual_0', color='red')
        s.plot_state_rviz(ref_actual_1, label='ref_actual_1', color=adjust_lightness('red', 0.5))
        s.plot_state_rviz(nearest_train_0, label='nearest_train_0', color='blue')
        s.plot_state_rviz(nearest_train_1, label='nearest_train_1', color=adjust_lightness('blue', 0.5))
        s.plot_action_rviz(nearest_train_0, nearest_train_1, label='nearest_train', color='blue')
        anim.step()


def _f(args):
    dataset, i = args
    traj = dataset[i]
    return get_actions(dataset.time_indexed_keys, traj)


def get_actions_list(dataset):
    with get_context("spawn").Pool() as p:
        n = len(dataset)
        result = list(tqdm(p.imap(_f, [(dataset, i) for i in range(n)], chunksize=512), total=n))

    actions_list = []
    example_indices = []
    for actions, example_idx in result:
        actions_list.append(actions)
        example_indices.append(example_idx)

    print("got actions from dataset")
    return actions_list, example_indices


def get_actions(time_indexed_keys, traj):
    s_0 = numpify(try_index_time(traj, time_indexed_keys, 0, False))
    left_gripper_0 = s_0['left_gripper']
    right_gripper_0 = s_0['right_gripper']
    before = np.stack([left_gripper_0, right_gripper_0])

    s_1 = numpify(try_index_time(traj, time_indexed_keys, 1, False))
    left_gripper_1 = s_1['left_gripper_position']
    right_gripper_1 = s_1['right_gripper_position']
    after = np.stack([left_gripper_1, right_gripper_1])
    origin = before[0]
    before_local = before - origin
    after_local = after - origin
    # actions = np.stack([before_local, after_local])
    actions = np.stack([before, after])

    return actions, traj['example_idx']


if __name__ == '__main__':
    main()
