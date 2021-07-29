#!/usr/bin/env python
import argparse
import pathlib

import numpy as np

import rospy
from arc_utilities import ros_init
from link_bot_classifiers.visualize_classifier_dataset import viz_compare_examples
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.pycommon import paths_from_json
from link_bot_pycommon.serialization import load_gzipped_pickle
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.filepath_tools import load_hjson
from moonshine.gpu_config import limit_gpu_mem
from rviz_voxelgrid_visuals_msgs.msg import VoxelgridStamped

limit_gpu_mem(None)


@ros_init.with_ros("viz_diversity")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('distances_filename', type=pathlib.Path)

    args = parser.parse_args()

    s = get_scenario("floating_rope")
    aug_env_pub = rospy.Publisher('aug_env', VoxelgridStamped, queue_size=10)
    data_env_pub = rospy.Publisher('data_env', VoxelgridStamped, queue_size=10)

    results = load_hjson(args.distances_filename)

    augfiles = paths_from_json(results['augfiles'])
    datafiles = paths_from_json(results['datafiles'])

    distances_matrix = np.ones([len(augfiles), len(datafiles)]) * 999
    for i in range(len(augfiles)):
        for j in range(len(datafiles)):
            k = f'{i}-{j}'
            if k in results:
                d = results[k]
                distances_matrix[i][j] = d

    def viz_pd(pd):
        if pd == 'plausibility':
            axis = 1
        elif pd == 'diversity':
            axis = 0
        else:
            raise NotImplementedError(pd)
        nearest_indices = np.argmin(distances_matrix, axis=axis)
        nearest_distances = np.min(distances_matrix, axis=axis)
        for data_i, (aug_j, d) in enumerate(zip(nearest_indices, nearest_distances)):
            # data_i is closest to aug_j, with distance d.
            aug_example = load_gzipped_pickle(augfiles[aug_j])
            data_example = load_gzipped_pickle(datafiles[data_i])

            viz_compare_examples(s, aug_example, data_example, aug_env_pub, data_env_pub)

    display_type = 'all'
    if display_type == 'all':
        for i in range(len(augfiles)):
            v = RvizAnimationController(len(datafiles))
            while not v.done:
                j = v.t()
                aug_example = load_gzipped_pickle(augfiles[i])
                data_example = load_gzipped_pickle(datafiles[j])
                viz_compare_examples(s, aug_example, data_example, aug_env_pub, data_env_pub)
                v.step()

    elif display_type == 'diversity':
        viz_pd(display_type)
    elif display_type == 'plausibility':
        viz_pd(display_type)


if __name__ == '__main__':
    main()
