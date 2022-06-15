#!/usr/bin/env python
import argparse
import pathlib
from time import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wandb
from matplotlib.animation import FuncAnimation
from torch.utils.data import DataLoader
from tqdm import tqdm

from arc_utilities import ros_init
from link_bot_data.new_dataset_utils import fetch_udnn_dataset
from link_bot_pycommon.load_wandb_model import load_model_artifact
from moonshine.grid_utils_tf import batch_point_to_idx
from moonshine.make_voxelgrid_inputs_torch import VoxelgridInfo
from moonshine.moonshine_utils import get_num_workers
from moonshine.robot_points_torch import RobotVoxelgridInfo
from moonshine.tfa_sdf import build_sdf_3d
from moonshine.torch_datasets_utils import my_collate, dataset_take
from state_space_dynamics.meta_udnn import UDNN
from state_space_dynamics.torch_dynamics_dataset import TorchDynamicsDataset


@ros_init.with_ros("anim_ile")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)
    parser.add_argument('checkpoint')

    args = parser.parse_args()

    dataset, data, xs, initial_errors_sorted, true_min_dists, root = compute_data(args)

    make_animation(args, data, xs, initial_errors_sorted, true_min_dists, root)


def compute_data(args, average_time=False, sort=False):
    dataset = TorchDynamicsDataset(fetch_udnn_dataset(args.dataset_dir), mode='notest')
    s = dataset.get_scenario()
    batch_size = 12
    dataset = dataset_take(dataset, 1)
    # dataset = dataset_shard(dataset, 200)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=my_collate, num_workers=get_num_workers(batch_size))
    root = pathlib.Path("results/anim_ile")
    root.mkdir(exist_ok=True, parents=True)

    ex0 = dataset[0]

    robot_info = RobotVoxelgridInfo(joint_positions_key='joint_positions',
                                    exclude_links=['leftgripper_link',
                                                   'leftgripper2_link',
                                                   'end_effector_left',
                                                   'rightgripper_link',
                                                   'rightgripper2_link',
                                                   'end_effector_right',
                                                   ])
    vg_info = VoxelgridInfo(h=ex0['env'].shape[0],
                            w=ex0['env'].shape[1],
                            c=ex0['env'].shape[2],
                            state_keys=[],
                            jacobian_follower=s.robot.jacobian_follower,
                            robot_info=robot_info,
                            include_robot_geometry=True,
                            scenario=s,
                            )

    api = wandb.Api()
    run = api.run(f'armlab/udnn/{args.checkpoint}')
    n_epochs = run.summary['epoch']
    data = []
    xs = []
    initial_errors_sorted = None
    true_min_dists = []
    sorted_indices = None
    epochs = [int(epoch) for epoch in np.linspace(130, n_epochs, 30)]
    epochs.insert(0, 0)
    for epoch in tqdm(epochs):
        model_at_epoch = load_model_artifact(args.checkpoint, UDNN, 'udnn', version=f'v{epoch}', user='armlab')
        error_at_epoch = []
        for i, inputs in enumerate(loader):
            outputs = model_at_epoch(inputs)
            batch_time_error = s.classifier_distance_torch(inputs, outputs)[:, 1:]  # skip t=0
            if average_time:
                batch_error = batch_time_error.mean(-1)  # average over time
                for error in batch_error:
                    error_at_epoch.append(float(error.detach().cpu().numpy()))
                    if epoch == 0:
                        xs.append(i)
            else:
                time = np.arange(batch_time_error.shape[1])
                batch_time = np.tile(time, [batch_time_error.shape[0], 1])
                batch_time = batch_time / batch_time_error.shape[1] + np.arange(batch_time_error.shape[0])[:, None]
                i_and_time = i + batch_time.flatten()
                error_at_epoch.extend(batch_time_error.flatten().detach().cpu().numpy().tolist())
                if epoch == 0:
                    xs.extend(i_and_time)
                    # add visualization of whether any of the true rope states are in collision

                    b, T, _ = inputs['rope'].shape
                    voxel_grids = vg_info.make_voxelgrid_inputs(inputs, inputs['env'], inputs['origin_point'], b, T)
                    voxel_grids = tf.clip_by_value(tf.reduce_sum(voxel_grids, axis=2), 0, 1)
                    # s.plot_environment_rviz({
                    #     'env':          voxel_grids[0, 0].numpy(),
                    #     'res':          inputs['res'][0],
                    #     'origin_point': inputs['origin_point'][0],
                    # })
                    sdf = build_sdf_3d(voxel_grids, inputs['res'])[:, 1:]
                    rope_points = inputs['rope'].reshape([b, T, 25, 3])[:, 1:]  # skip t=0
                    batch_sdf_indices = batch_point_to_idx(rope_points, inputs['res'], inputs['origin_point'])
                    sdf_values = tf.gather_nd(sdf, batch_sdf_indices, batch_dims=2)
                    true_min_dist = tf.reduce_min(sdf_values, -1)  # [b, T]
                    true_min_dist = true_min_dist.numpy().flatten().tolist()
                    true_min_dists.extend(true_min_dist)

        error_at_epoch = np.array(error_at_epoch)
        if epoch == 0:
            if sort:
                sorted_indices = np.argsort(error_at_epoch)
                error_at_epoch_sorted = error_at_epoch[sorted_indices]
            else:
                error_at_epoch_sorted = error_at_epoch
            initial_errors_sorted = error_at_epoch_sorted
        else:
            if sort:
                error_at_epoch_sorted = error_at_epoch[sorted_indices]
            else:
                error_at_epoch_sorted = error_at_epoch

        row = [epoch, error_at_epoch_sorted]
        data.append(row)
    return dataset, data, xs, initial_errors_sorted, true_min_dists, root


def make_animation(args, data, xs, initial_errors_sorted, true_min_dists, root):
    fig = plt.figure(figsize=(10, 4))
    ax = plt.gca()
    ax.set_xlabel("training data, sorted by time")
    ax.set_ylabel("error")
    ax.set_ylim(bottom=9e-3, top=1.5)
    ax.set_yscale("log")
    s = 2
    scatt = ax.scatter(xs, np.ones(len(data[0][1])), label='current', s=s)
    ax.scatter(xs, initial_errors_sorted, label='initial error', s=s)
    ax.axhline(y=0.08, color='k', linestyle='--')
    ax.plot(xs, true_min_dists, label='true min dist to obs/robot', color='m')
    plt.legend(loc='upper right')

    def _viz_epoch(frame_idx):
        row = data[frame_idx]
        epoch, error_at_epoch = row
        scatt.set_offsets(np.stack([xs, error_at_epoch], 1))
        ax.set_title(f"{epoch=}")

    anim = FuncAnimation(fig=fig, func=_viz_epoch, frames=len(data))
    filename = root / f"anim_ile-{args.dataset_dir}-{args.checkpoint}-{int(time())}.gif"
    print(f"saving {filename.as_posix()}")
    anim.save(filename.as_posix(), fps=5)
    plt.close()
    print("done")


if __name__ == '__main__':
    main()
