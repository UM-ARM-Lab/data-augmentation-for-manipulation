#!/usr/bin/env python
import argparse
import pathlib

import numpy as np

import rospy
from arc_utilities import ros_init
from link_bot_pycommon import grid_utils
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.grid_utils import environment_to_vg_msg
from link_bot_pycommon.serialization import load_gzipped_pickle
from moonshine.filepath_tools import load_hjson
from moonshine.gpu_config import limit_gpu_mem
from rviz_voxelgrid_visuals_msgs.msg import VoxelgridStamped

limit_gpu_mem(None)


@ros_init.with_ros("viz_diversity")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('augdir', type=pathlib.Path)
    parser.add_argument('datadir', type=pathlib.Path)
    parser.add_argument('distances_filename', type=pathlib.Path)

    args = parser.parse_args()

    augfiles = list(args.augdir.glob("*.pkl.gz"))
    datafiles = list(args.datadir.glob("*.pkl.gz"))

    distances = load_hjson(args.distances_filename)
    distances_matrix = np.ones([len(augfiles), len(datafiles)]) * 999
    for i in range(len(augfiles)):
        for j in range(len(datafiles)):
            k = f'{i}-{j}'
            if k in distances:
                d = distances[k]
                distances_matrix[i][j] = d

    nearest_indices = np.argmin(distances_matrix, axis=0)
    nearest_distances = np.min(distances_matrix, axis=0)

    s = get_scenario("floating_rope")
    aug_env_pub = rospy.Publisher('aug_env', VoxelgridStamped, queue_size=10)
    data_env_pub = rospy.Publisher('data_env', VoxelgridStamped, queue_size=10)

    for data_i, (aug_j, d) in enumerate(zip(nearest_indices, nearest_distances)):
        # data_i is closest to aug_j, with distance d.
        aug_example = load_gzipped_pickle(augfiles[aug_j])
        data_example = load_gzipped_pickle(datafiles[data_i])

        def plot_example(e, label, env_pub):
            state_before = {
                'rope':            e['rope'][0],
                'joint_positions': e['joint_positions'][0],
            }
            state_after = {
                'rope':            e['rope'][1],
                'joint_positions': e['joint_positions'][1],
            }
            s.plot_state_rviz(state_before, label=label + '_before')
            s.plot_state_rviz(state_after, label=label + '_after')
            env = {
                'env':          e['env'],
                'res':          0.02,
                'origin_point': np.array([1.0, 0, 0]),
            }
            s.plot_environment_rviz(env)

            frame = 'env_vg'
            env_msg = environment_to_vg_msg(env, frame=frame)
            env_pub.publish(env_msg)
            grid_utils.send_voxelgrid_tf_origin_point_res(s.tf.tf_broadcaster,
                                                          env['origin_point'],
                                                          env['res'],
                                                          frame=frame)
            s.tf.send_transform(env['origin_point'], [0, 0, 0, 1], 'world', child='origin_point')

        plot_example(data_example, 'data', data_env_pub)
        plot_example(aug_example, 'aug', aug_env_pub)


if __name__ == '__main__':
    main()
