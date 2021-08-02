#!/usr/bin/env python
import argparse
import pathlib
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import rospy
from arc_utilities import ros_init
from link_bot_classifiers.pd_distances_utils import too_far
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
        #
        # viz_compare_examples(aug_viz_info.s, aug_example, data_example, aug_viz_info.aug_env_pub,
        #                      aug_viz_info.data_env_pub)
        pass


def _stem(p):
    return p.name.split('.')[0]


def format_distances(aug_viz_info: AugVizInfo, results_dir: pathlib.Path, space_idx: int):
    results_filenames = list(results_dir.glob("*.pkl.gz"))
    n_aug = max([int(_stem(filename).split('-')[0]) for filename in results_filenames]) + 1
    n_data = max([int(_stem(filename).split('-')[1]) for filename in results_filenames]) + 1
    shape = [n_aug, n_data]
    distances_matrix = np.ones(shape) * too_far[space_idx]
    aug_examples_matrix = np.empty(shape, dtype=object)
    data_examples_matrix = np.empty(shape, dtype=object)
    for results_filename in tqdm(results_filenames):
        result = load_gzipped_pickle(results_filename)
        # with results_filename.open("rb") as f:
        #     result = pickle.load(f)
        distances = result['distance']
        i = int(_stem(results_filename).split('-')[0])
        j = int(_stem(results_filename).split('-')[1])

        distances_matrix[i][j] = distances[space_idx]
        aug_examples_matrix[i][j] = result['aug_example']
        data_examples_matrix[i][j] = result['data_example']
    return aug_examples_matrix, data_examples_matrix, distances_matrix


@ros_init.with_ros("viz_diversity")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', type=pathlib.Path)
    parser.add_argument('display_type', choices=['data_all', 'aug_all', 'both', 'plausibility', 'diversity'])
    parser.add_argument('space', choices=['rope', 'robot', 'env'])

    args = parser.parse_args()

    s = get_scenario("dual_arm_rope_sim_val")
    aug_env_pub = rospy.Publisher('aug_env', VoxelgridStamped, queue_size=10)
    data_env_pub = rospy.Publisher('data_env', VoxelgridStamped, queue_size=10)

    aug_viz_info = AugVizInfo(s=s, aug_env_pub=aug_env_pub, data_env_pub=data_env_pub)

    if args.space == 'rope':
        space_idx = 0
    elif args.space == 'robot':
        space_idx = 3
    elif args.space == 'env':
        space_idx = 4
    else:
        raise NotImplementedError(args.space)

    aug_examples_matrix, data_examples_matrix, distances_matrix = format_distances(aug_viz_info=aug_viz_info,
                                                                                   results_dir=args.results_dir,
                                                                                   space_idx=space_idx)

    def get_first_non_none(m):
        for i, m_i in enumerate(m):
            if m_i is not None:
                return i, m_i
        return 0, None

    if args.display_type == 'aug_all':
        for i in range(aug_examples_matrix.shape[0]):
            max_j, aug_example = get_first_non_none(aug_examples_matrix[i])
            if max_j == 0:
                print("no close examples")
                continue
            distances_for_aug_i = distances_matrix[i]
            sorted_indices = np.argsort(distances_for_aug_i)
            sorted_data_examples = np.take(data_examples_matrix[i], sorted_indices)
            distances_for_aug_i_sorted = np.take(distances_for_aug_i, sorted_indices)
            print(distances_for_aug_i_sorted[0])
            v = RvizAnimationController(n_time_steps=max_j)
            while not v.done:
                sorted_j = v.t()
                data_example = sorted_data_examples[sorted_j]
                d = distances_for_aug_i_sorted[sorted_j]
                if aug_example is not None and data_example is not None:
                    viz_compare_examples(s, aug_example, data_example, aug_env_pub, data_env_pub)
                    s.plot_error_rviz(d)
                v.step()
    elif args.display_type == 'data_all':
        for j in range(aug_examples_matrix.shape[1]):
            max_i, data_example = get_first_non_none(data_examples_matrix[:, j])  # 0 works because they're all the same
            if max_i == 0:
                print("no close examples")
                continue
            distances_for_data_j = distances_matrix[:, j]
            sorted_indices = np.argsort(distances_for_data_j)
            sorted_aug_examples = np.take(aug_examples_matrix[:, j], sorted_indices)
            distances_for_data_j_sorted = np.take(distances_for_data_j, sorted_indices)
            print(distances_for_data_j_sorted[0])
            v = RvizAnimationController(n_time_steps=max_i)
            while not v.done:
                sorted_j = v.t()
                aug_example = sorted_aug_examples[sorted_j]
                d = distances_for_data_j_sorted[sorted_j]
                if aug_example is not None and data_example is not None:
                    viz_compare_examples(s, aug_example, data_example, aug_env_pub, data_env_pub)
                    s.plot_error_rviz(d)
                v.step()
    elif args.display_type == 'both':
        diversities = []
        for j in range(aug_examples_matrix.shape[1]):
            distances_for_data_j = distances_matrix[:, j]
            sorted_indices = np.argsort(distances_for_data_j)
            distances_for_data_j_sorted = np.take(distances_for_data_j, sorted_indices)
            diversity = 1 / distances_for_data_j_sorted[0]
            diversities.append(diversity)
        plausibilities = []
        for i in range(aug_examples_matrix.shape[0]):
            distances_for_aug_i = distances_matrix[i]
            sorted_indices = np.argsort(distances_for_aug_i)
            distances_for_aug_i_sorted = np.take(distances_for_aug_i, sorted_indices)
            plausibility = 1 / distances_for_aug_i_sorted[0]
            plausibilities.append(plausibility)

        print(f'\tP: {np.mean(plausibilities):.3f}, {np.std(plausibilities):.3f}, {len(plausibilities)}')
        print(f'\tD: {np.mean(diversities):.3f}, {np.std(diversities):.3f}, {len(diversities)}')
        bins = 200
        fig, axes = plt.subplots(1, 2)
        axes[0].hist(plausibilities, label='plausibility', bins=bins, alpha=0.5)
        axes[0].legend()
        axes[0].set_xlabel("1 / distance to nearest")
        axes[0].set_ylabel("count")
        axes[1].hist(diversities, label='diversity', bins=bins, alpha=0.5)
        axes[1].legend()
        axes[1].set_xlabel("1 / distance to nearest")
        axes[0].set_ylabel("count")
        fig.suptitle(f'{args.results_dir.name} {args.space}')
        plt.savefig(args.results_dir / f'{args.space}.png')
        plt.show()
    else:
        viz_pd(aug_viz_info, args.display_type, distances_matrix)


if __name__ == '__main__':
    main()
