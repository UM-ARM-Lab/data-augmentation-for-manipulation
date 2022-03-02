#!/usr/bin/env python
import argparse
import pathlib
from typing import Dict

import numpy as np
import tensorflow as tf

from arc_utilities import ros_init
from arc_utilities.filesystem_utils import no_overwrite_path
from link_bot_data.modify_dataset import modify_dataset2
from link_bot_data.new_base_dataset import NewBaseDatasetLoader
from link_bot_data.robot_points import batch_robot_state_to_transforms, batch_transform_robot_points, RobotVoxelgridInfo
from link_bot_data.split_dataset import split_dataset_via_files
from link_bot_pycommon.collision_checking import batch_in_collision_tf_3d
from link_bot_pycommon.grid_utils_np import extent_to_env_shape
from moonshine.gpu_config import limit_gpu_mem
from moonshine.raster_3d import points_to_voxel_grid_res_origin_point_batched

limit_gpu_mem(None)


@ros_init.with_ros("heuristic_data_weights")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')

    args = parser.parse_args()

    outdir = args.dataset_dir.parent / f"{args.dataset_dir.name}+heuristic-weights"
    outdir = no_overwrite_path(outdir)
    print(outdir)
    dataset = NewBaseDatasetLoader([args.dataset_dir])
    scenario = dataset.get_scenario()
    robot_points_path = pathlib.Path("robot_points_data/val_high_res/robot_points.pkl")
    robot_info = RobotVoxelgridInfo('joint_positions', robot_points_path)

    MAX_ROPE_LENGTH = 0.774  # max length in unconstrained_alt_1646066051_8d6b29f08f_750

    def check_in_collision(environment, points, inflation):
        in_collision, inflated_env = batch_in_collision_tf_3d(environment=environment,
                                                              points=points,
                                                              inflate_radius_m=inflation)
        in_collision = in_collision.numpy()
        return in_collision

    def make_robot_voxelgrid(example, origin_point):
        # batch here means time
        batch_size = example['time_idx'].shape[0]
        h, w, c = extent_to_env_shape(example['extent'], robot_info.res)
        joint_names = example['joint_names']
        joint_positions = example['joint_positions']
        link_to_robot_transforms = batch_robot_state_to_transforms(scenario.robot.jacobian_follower,
                                                                   joint_names,
                                                                   joint_positions,
                                                                   robot_info.link_names)
        robot_points = batch_transform_robot_points(link_to_robot_transforms, robot_info, batch_size)
        n_robot_points = robot_points.shape[1]
        flat_batch_indices = tf.repeat(tf.range(batch_size, dtype=tf.int64), n_robot_points, axis=0)
        flat_points = tf.reshape(robot_points, [-1, 3])
        flat_points.set_shape([n_robot_points * batch_size, 3])
        flat_res = tf.repeat(robot_info.res, n_robot_points * batch_size, axis=0)
        flat_origin_point = tf.repeat(tf.expand_dims(origin_point, 0), n_robot_points * batch_size, axis=0)
        robot_voxelgrid = points_to_voxel_grid_res_origin_point_batched(flat_batch_indices,
                                                                        flat_points,
                                                                        flat_res,
                                                                        flat_origin_point,
                                                                        h,
                                                                        w,
                                                                        c,
                                                                        batch_size)
        return robot_voxelgrid

    def _process_example(dataset, example: Dict):
        points = scenario.state_to_points_for_cc(example)
        in_collision = check_in_collision(example, points, float(tf.squeeze(example['res'])))
        d = tf.linalg.norm(example['right_gripper'] - example['left_gripper'], axis=-1)
        too_far = d > 0.55  # copied from floating_rope.hjson data collection params, max_distance_between_grippers

        robot_voxel_grid = make_robot_voxelgrid(example, example['origin_point'])
        time = example['time_idx'].shape[0]
        robot_in_collision = []
        for t in range(time):
            robot_as_env_t = {
                'env':          robot_voxel_grid[t],
                'res':          robot_info.res,
                'origin_point': example['origin_point'],
            }
            robot_in_collision_t = check_in_collision(robot_as_env_t, points[t], robot_info.res * 0.7)
            robot_in_collision.append(robot_in_collision_t)
            # if robot_in_collision_t:
            #     scenario.plot_environment_rviz(robot_as_env_t)
            #     scenario.plot_points_rviz(tf.reshape(points[t], [-1, 3]).numpy(), label='cc', scale=0.005)

        rope_points = example['rope'].reshape([10, 25, 3])
        rope_length = np.sum(np.linalg.norm(rope_points[:, :-1] - rope_points[:, 1:], axis=-1), axis=-1)
        too_long = rope_length > MAX_ROPE_LENGTH

        weight = 1 - np.logical_or.reduce((in_collision, robot_in_collision, too_far, too_long)).astype(np.float32)
        weight_padded = np.concatenate((weight, [1]))
        weight = np.logical_and(weight_padded[:-1], weight_padded[1:]).astype(np.float32)
        example['metadata']['weight'] = weight
        yield example

    hparams_update = {}

    modify_dataset2(dataset_dir=args.dataset_dir,
                    dataset=dataset,
                    outdir=outdir,
                    process_example=_process_example,
                    hparams_update=hparams_update,
                    save_format='pkl')
    split_dataset_via_files(outdir, 'pkl')
    print(outdir)


if __name__ == '__main__':
    main()
