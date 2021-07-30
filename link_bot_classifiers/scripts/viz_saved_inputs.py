#!/usr/bin/env python
import argparse
import pathlib
from dataclasses import dataclass
from typing import List, Dict

import numpy as np

import rospy
from arc_utilities import ros_init
from link_bot_classifiers.visualize_classifier_dataset import viz_compare_example
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from link_bot_pycommon.serialization import load_gzipped_pickle
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.gpu_config import limit_gpu_mem
from moonshine.indexing import index_dict_of_batched_tensors_tf
from rviz_voxelgrid_visuals_msgs.msg import VoxelgridStamped

limit_gpu_mem(None)


@dataclass
class AugVizInfo:
    s: ScenarioWithVisualization
    augfiles: List[pathlib.Path]
    datafiles: List[pathlib.Path]
    aug_env_pub: rospy.Publisher
    data_env_pub: rospy.Publisher


def format_distances(aug_viz_info: AugVizInfo, results: Dict):
    distances_matrix = np.ones([len(aug_viz_info.augfiles), len(aug_viz_info.datafiles)]) * 999
    aug_examples_matrix = np.empty([len(aug_viz_info.augfiles), len(aug_viz_info.datafiles)], dtype=object)
    data_examples_matrix = np.empty([len(aug_viz_info.augfiles), len(aug_viz_info.datafiles)], dtype=object)
    for i in range(len(aug_viz_info.augfiles)):
        for j in range(len(aug_viz_info.datafiles)):
            k = f'{i}-{j}'
            if k in results:
                d = results[k]
                distances_matrix[i][j] = d['distance'][0]
                aug_examples_matrix[i][j] = d['aug_example']
                data_examples_matrix[i][j] = d['data_example']
    return aug_examples_matrix, data_examples_matrix, distances_matrix


@ros_init.with_ros("viz_saved_inputs")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('saved_inputs', type=pathlib.Path)

    args = parser.parse_args()

    s = get_scenario("dual_arm_rope_sim_val")
    env_pub = rospy.Publisher('env', VoxelgridStamped, queue_size=10)

    filenames = sorted(list(args.saved_inputs.glob("*.pkl.gz")))

    r = RvizAnimationController(n_time_steps=len(filenames))
    while not r.done and not rospy.is_shutdown():
        t = r.t()
        filename_t = filenames[t]
        print(filename_t)
        example_t = load_gzipped_pickle(filename_t)
        viz_compare_example(s, example_t, '', env_pub, color='m')
        r.step()


if __name__ == '__main__':
    main()
