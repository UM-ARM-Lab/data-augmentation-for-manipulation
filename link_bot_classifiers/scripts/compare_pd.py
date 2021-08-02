#!/usr/bin/env python
import numpy as np
import scipy.stats
import argparse
import pathlib

from colorama import Fore

from arc_utilities import ros_init
from link_bot_classifiers.pd_distances_utils import format_distances, space_to_idx, \
    compute_plausibility, compute_diversity
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(None)


def print_test(name1, d1, name2, d2):
    print(f'{name1:.30s}  mean: {np.mean(d1):.3f}, std: {np.std(d1):.3f}')
    print(f'{name2:.30s}  mean: {np.mean(d2):.3f}, std: {np.std(d2):.3f}')
    print(f'p={scipy.stats.ttest_ind(d1, d2)[1]:.4f}')


@ros_init.with_ros("compare_pd")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('d1', type=pathlib.Path)
    parser.add_argument('d2', type=pathlib.Path)

    args = parser.parse_args()

    for space in ['rope', 'robot', 'env']:
        print(Fore.CYAN + space + Fore.RESET)
        space_idx = space_to_idx(space)

        _, _, distances_matrix1 = format_distances(results_dir=args.d1, space_idx=space_idx)
        _, _, distances_matrix2 = format_distances(results_dir=args.d2, space_idx=space_idx)

        d1 = compute_diversity(distances_matrix1)
        d2 = compute_diversity(distances_matrix2)
        print(Fore.GREEN + 'Diversity' + Fore.RESET)
        print_test(args.d1.name, d1, args.d2.name, d2)

        p1 = compute_plausibility(distances_matrix1)
        p2 = compute_plausibility(distances_matrix2)
        print(Fore.GREEN + 'Plausibility' + Fore.RESET)
        print_test(args.d1.name, p1, args.d2.name, p2)

        print()
        print()


if __name__ == '__main__':
    main()
