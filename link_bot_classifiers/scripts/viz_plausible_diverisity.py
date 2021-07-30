#!/usr/bin/env python
import argparse
import pathlib
import pickle
from dataclasses import dataclass
from typing import List, Dict

import numpy as np

import rospy
from arc_utilities import ros_init
from link_bot_classifiers.visualize_classifier_dataset import viz_compare_examples
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from link_bot_pycommon.serialization import load_gzipped_pickle
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.gpu_config import limit_gpu_mem
from rviz_voxelgrid_visuals_msgs.msg import VoxelgridStamped

limit_gpu_mem(None)


@dataclass
class AugVizInfo:
    s: ScenarioWithVisualization
    augfiles: List[pathlib.Path]
    datafiles: List[pathlib.Path]
    aug_env_pub: rospy.Publisher
    data_env_pub: rospy.Publisher


def viz_pd(aug_viz_info: AugVizInfo, pd, distances_matrix):
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
        # aug_example = load_gzipped_pickle(aug_viz_info.augfiles[aug_j])
        # data_example = load_gzipped_pickle(aug_viz_info.datafiles[data_i])

        viz_compare_examples(aug_viz_info.s, aug_example, data_example, aug_viz_info.aug_env_pub,
                             aug_viz_info.data_env_pub)


def format_distances(aug_viz_info: AugVizInfo, results_dir: pathlib.Path):
    results_filenames = results_dir.glob("*.pkl")
    distances_matrix = np.ones([len(aug_viz_info.augfiles), len(aug_viz_info.datafiles)]) * 999
    aug_examples_matrix = np.empty([len(aug_viz_info.augfiles), len(aug_viz_info.datafiles)], dtype=object)
    data_examples_matrix = np.empty([len(aug_viz_info.augfiles), len(aug_viz_info.datafiles)], dtype=object)
    for results_filename in results_filenames:
        with results_filename.open("rb") as f:
            result = pickle.load(f)
        distances = result['distance']
        distances_matrix[i][j] = distances[0]
        aug_examples_matrix[i][j] = result['aug_example']
        data_examples_matrix[i][j] = result['data_example']
    return aug_examples_matrix, data_examples_matrix, distances_matrix


@ros_init.with_ros("viz_diversity")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', type=pathlib.Path)

    args = parser.parse_args()

    s = get_scenario("dual_arm_rope_sim_val")
    aug_env_pub = rospy.Publisher('aug_env', VoxelgridStamped, queue_size=10)
    data_env_pub = rospy.Publisher('data_env', VoxelgridStamped, queue_size=10)

    aug_viz_info = AugVizInfo(s=s,
                              aug_env_pub=aug_env_pub,
                              data_env_pub=data_env_pub,
                              augfiles=[],
                              datafiles=[])

    aug_examples_matrix, data_examples_matrix, distances_matrix = format_distances(aug_viz_info=aug_viz_info,
                                                                                   results_dir=args.results_dir)

    display_type = 'data_all'
    if display_type == 'aug_all':
        for i in range(aug_examples_matrix.shape[0]):
            # aug_example = load_gzipped_pickle(augfiles[i])
            aug_example = aug_examples_matrix[i]
            distances_for_aug_i = distances_matrix[i]
            sorted_indices = np.argsort(distances_for_aug_i)
            sorted_data_examples = np.take(data_examples_matrix[i], sorted_indices)
            distances_for_aug_i_sorted = np.take(distances_for_aug_i, sorted_indices)
            v = RvizAnimationController(n_time_steps=aug_examples_matrix.shape[1])
            while not v.done:
                sorted_j = v.t()
                # data_example = load_gzipped_pickle(sorted_datafiles[sorted_j])
                data_example = data_examples_matrix[sorted_j]
                d = distances_for_aug_i_sorted[sorted_j]
                viz_compare_examples(s, aug_example, data_example, aug_env_pub, data_env_pub)
                s.plot_error_rviz(d)
                v.step()
    elif display_type == 'data_all':
        for j in range(aug_examples_matrix.shape[1]):
            data_example = data_examples_matrix[j]
            distances_for_data_j = distances_matrix[:, j]
            sorted_indices = np.argsort(distances_for_data_j)
            sorted_aug_examples = np.take(aug_examples_matrix[j], sorted_indices)
            distances_for_aug_i_sorted = np.take(distances_for_data_j, sorted_indices)
            v = RvizAnimationController(n_time_steps=aug_examples_matrix.shape[0])
            while not v.done:
                sorted_j = v.t()
                aug_example = sorted_aug_examples[sorted_j]
                d = distances_for_aug_i_sorted[sorted_j]
                viz_compare_examples(s, aug_example, data_example, aug_env_pub, data_env_pub)
                s.plot_error_rviz(d)
                v.step()
    else:
        viz_pd(aug_viz_info, display_type, distances_matrix)


if __name__ == '__main__':
    main()
