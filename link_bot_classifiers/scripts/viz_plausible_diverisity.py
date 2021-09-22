#!/usr/bin/env python
import argparse
import pathlib

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

import rospy
from arc_utilities import ros_init
from link_bot_classifiers.pd_distances_utils import format_distances, get_first, space_to_idx, \
    compute_diversity, compute_plausibility
from link_bot_classifiers.visualize_classifier_dataset import viz_compare_examples
from link_bot_data.dataset_utils import add_predicted
from link_bot_pycommon.get_scenario import get_scenario
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.gpu_config import limit_gpu_mem
from rviz_voxelgrid_visuals_msgs.msg import VoxelgridStamped

limit_gpu_mem(None)


@ros_init.with_ros("viz_diversity")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('distances_dir', type=pathlib.Path, help='the result of compute distances')
    parser.add_argument('display_type',
                        choices=['both',
                                 'plausibility',
                                 'diversity',
                                 'diversity_negatives',
                                 'plausibility_all',
                                 'diversity_all'])
    parser.add_argument('space', choices=['pred_rope', 'rope', 'robot', 'env'])

    args = parser.parse_args()

    s = get_scenario("dual_arm_rope_sim_val")
    aug_env_pub = rospy.Publisher('aug_env', VoxelgridStamped, queue_size=10)
    data_env_pub = rospy.Publisher('data_env', VoxelgridStamped, queue_size=10)

    use_predicted = (args.space == 'pred_rope')
    space_idx = space_to_idx(args.space)

    aug_examples_matrix, data_examples_matrix, distances_matrix = format_distances(results_dir=args.distances_dir,
                                                                                   space_idx=space_idx)

    def viz_diversity_examples(v: RvizAnimationController):
        while not v.done:
            j = v.t()
            max_i, data_example = get_first(data_examples_matrix[:, j])
            label = get_data_is_close(data_example)
            if max_i == -1:
                print("no close examples")
            else:
                distances_for_data_j = distances_matrix[:, j]
                best_idx = np.argmin(distances_for_data_j)
                aug_example = aug_examples_matrix[best_idx, j]
                aug_label = aug_example['is_close'][1]
                best_d = distances_for_data_j[best_idx]
                print(f"{j} {best_d:.3f} data_label={label} aug_label={aug_label}")
                viz_compare_examples(s, aug_example, data_example, aug_env_pub, data_env_pub)
                if aug_example is not None and data_example is not None:
                    s.plot_error_rviz(best_d)
            v.step()

    def get_data_is_close(data_example):
        if 'is_close' not in data_example:
            predicted = {'rope': data_example[add_predicted('rope')]}
            actual = {'rope': data_example['rope']}
            data_example['is_close'] = np.float32(
                s.compute_label(actual, predicted, labeling_params={'threshold': 0.05}))
        label = data_example['is_close'][1]
        return label

    if args.display_type == 'plausibility':
        v = RvizAnimationController(n_time_steps=aug_examples_matrix.shape[0])
        while not v.done:
            i = v.t()
            max_j, aug_example = get_first(aug_examples_matrix[i])
            if max_j == -1:
                print("no close examples")
            else:
                distances_for_aug_i = distances_matrix[i]
                best_idx = np.argmin(distances_for_aug_i)
                best_d = distances_for_aug_i[best_idx]
                data_example = data_examples_matrix[i, best_idx]
                print(i, best_d)
                viz_compare_examples(s, aug_example, data_example, aug_env_pub, data_env_pub)
                if aug_example is not None and data_example is not None:
                    s.plot_error_rviz(best_d)
            v.step()
    elif args.display_type == 'diversity_negatives':
        data_is_negative_indices = []
        for j, r in enumerate(data_examples_matrix[0]):
            if r is not None:
                if r['is_close'][1] == 0:
                    data_is_negative_indices.append(j)
        v = RvizAnimationController(time_steps=data_is_negative_indices)
        viz_diversity_examples(v)
    elif args.display_type == 'diversity':
        v = RvizAnimationController(n_time_steps=aug_examples_matrix.shape[1])
        viz_diversity_examples(v)
    elif args.display_type == 'plausibility_all':
        for i in range(aug_examples_matrix.shape[0]):
            max_j, aug_example = get_first(aug_examples_matrix[i])
            if max_j == -1:
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
            max_i, data_example = get_first(data_examples_matrix[:, j])
            if max_i == -1:
                print("no close examples")
                continue
            distances_for_data_j = distances_matrix[:, j]
            sorted_indices = np.argsort(distances_for_data_j)
            sorted_aug_examples = np.take(aug_examples_matrix[:, j], sorted_indices)
            distances_for_data_j_sorted = np.take(distances_for_data_j, sorted_indices)
            print(distances_for_data_j_sorted[0])
            v = RvizAnimationController(n_time_steps=distances_for_data_j.shape[0])
            while not v.done:
                sorted_j = v.t()
                aug_example = sorted_aug_examples[sorted_j]
                aug_label = aug_example['is_close'][1]
                label = get_data_is_close(data_example)
                d = distances_for_data_j_sorted[sorted_j]
                print(f"{sorted_j} distance={d} data_label={label} aug_label={aug_label}")
                if aug_example is not None and data_example is not None:
                    viz_compare_examples(s, aug_example, data_example, aug_env_pub, data_env_pub, use_predicted)
                    s.plot_error_rviz(d)
                v.step()
    elif args.display_type == 'both':
        diversities = compute_diversity(distances_matrix)
        plausibilities = compute_plausibility(distances_matrix)

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
        fig.suptitle(f'{args.distances_dir.name} {args.space}')
        plt.savefig(args.distances_dir / f'{args.space}.png')
        plt.show()
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    main()
