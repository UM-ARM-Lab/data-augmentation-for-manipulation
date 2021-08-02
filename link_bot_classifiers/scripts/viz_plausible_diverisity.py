#!/usr/bin/env python
import argparse
import pathlib
from dataclasses import dataclass
from functools import lru_cache

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
from moonshine.filepath_tools import load_hjson
from moonshine.gpu_config import limit_gpu_mem
from rviz_voxelgrid_visuals_msgs.msg import VoxelgridStamped

limit_gpu_mem(None)


def _stem(p):
    return p.name.split('.')[0]


class LazyMatrix:

    def __init__(self, shape):
        self.shape = shape
        self.m = np.empty(shape, dtype=object)

    @lru_cache
    def __getitem__(self, indices):
        # results_filenames = list(results_dir.glob("*.pkl.gz"))
        # for results_filename in tqdm(results_filenames):
        #     result = load_gzipped_pickle(results_filename)
        #     distances = result['distance']
        #     i = int(_stem(results_filename).split('-')[0])
        #     j = int(_stem(results_filename).split('-')[1])
        #     aug_examples_matrix[i][j] = result['aug_example']
        #     data_examples_matrix[i][j] = result['data_example']
        #
        #     distances_matrix[i][j] = distances[space_idx]
        if isinstance(indices, int):
            pass
        elif isinstance(indices, tuple):
            pass
        else:
            raise NotImplementedError(f'unsupported index {indices} of type {type(indices)}')


def format_distances(results_dir: pathlib.Path, space_idx: int):
    logfilename = results_dir / 'logfile.hjson'
    log = load_hjson(logfilename)
    log.pop("augfiles")
    log.pop("datafiles")
    log.pop("weights")
    n_aug = max([int(k.split('-')[0]) for k in log.keys()]) + 1
    n_data = max([int(k.split('-')[1]) for k in log.keys()]) + 1
    distances_matrix = np.ones([n_aug, n_data, 5]) * too_far[space_idx]
    aug_examples_matrix = LazyMatrix([n_aug, n_data])
    data_examples_matrix = LazyMatrix([n_aug, n_data])

    for k, d in log.items():
        aug_i, data_j = k.split('-')
        aug_i = int(aug_i)
        data_j = int(data_j)
        distances_matrix[aug_i][data_j] = d[space_idx]

    return aug_examples_matrix, data_examples_matrix, distances_matrix


@ros_init.with_ros("viz_diversity")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', type=pathlib.Path)
    parser.add_argument('display_type',
                        choices=['both', 'plausibility', 'diversity', 'plausibility_all', 'diversity_all'])
    parser.add_argument('space', choices=['rope', 'robot', 'env'])

    args = parser.parse_args()

    s = get_scenario("dual_arm_rope_sim_val")
    aug_env_pub = rospy.Publisher('aug_env', VoxelgridStamped, queue_size=10)
    data_env_pub = rospy.Publisher('data_env', VoxelgridStamped, queue_size=10)

    if args.space == 'rope':
        space_idx = 0
    elif args.space == 'robot':
        space_idx = 3
    elif args.space == 'env':
        space_idx = 4
    else:
        raise NotImplementedError(args.space)

    aug_examples_matrix, data_examples_matrix, distances_matrix = format_distances(results_dir=args.results_dir,
                                                                                   space_idx=space_idx)

    def get_first_non_none(m):
        for i, m_i in enumerate(m):
            if m_i is not None:
                return i, m_i
        return 0, None

    if args.display_type == 'plausibility':
        v = RvizAnimationController(n_time_steps=aug_examples_matrix.shape[0])
        while not v.done:
            i = v.t()
            max_j, aug_example = get_first_non_none(aug_examples_matrix[i])
            if max_j == 0:
                print("no close examples")
            else:
                distances_for_aug_i = distances_matrix[i]
                best_idx = np.argmin(distances_for_aug_i)
                best_d = distances_for_aug_i[best_idx]
                data_example = data_examples_matrix[i][best_idx]
                print(i, best_d)
                if aug_example is not None and data_example is not None:
                    viz_compare_examples(s, aug_example, data_example, aug_env_pub, data_env_pub)
                    s.plot_error_rviz(best_d)
            v.step()
    elif args.display_type == 'diversity':
        v = RvizAnimationController(n_time_steps=aug_examples_matrix.shape[1])
        while not v.done:
            j = v.t()
            max_i, data_example = get_first_non_none(data_examples_matrix[:, j])
            if max_i == 0:
                print("no close examples")
            else:
                distances_for_data_j = distances_matrix[:, j]
                best_idx = np.argmin(distances_for_data_j)
                aug_example = aug_examples_matrix[best_idx, j]
                best_d = distances_for_data_j[best_idx]
                print(j, best_d)
                if aug_example is not None and data_example is not None:
                    viz_compare_examples(s, aug_example, data_example, aug_env_pub, data_env_pub)
                    s.plot_error_rviz(best_d)
            v.step()
    elif args.display_type == 'plausibility_all':
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
    elif args.display_type == 'diversity_all':
        for j in range(aug_examples_matrix.shape[1]):
            max_i, data_example = get_first_non_none(data_examples_matrix[:, j])
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
        raise NotImplementedError()


if __name__ == '__main__':
    main()
