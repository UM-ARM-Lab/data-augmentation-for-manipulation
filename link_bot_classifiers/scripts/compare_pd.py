#!/usr/bin/env python
import argparse
import pathlib
from typing import List

import numpy as np
from IPython.utils.text import long_substr
from colorama import Fore

from arc_utilities import ros_init
from link_bot_classifiers.pd_distances_utils import format_distances, space_to_idx, \
    compute_plausibility, compute_diversity
from link_bot_pycommon.metric_utils import dict_to_pvalue_table
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(None)


def shorten(name):
    return name.replace('-', '').replace('_', '').replace('/', '').replace(' ', '')


def make_useful_names(names: List[str]):
    useful_names = names.copy()
    while True:
        s = long_substr(useful_names)
        s = s.lstrip('-_')
        if len(s) <= 1:
            break
        useful_names = [n.replace(s, '') for n in useful_names]
    useful_names = [n.strip('-_') for n in useful_names]
    return useful_names


@ros_init.with_ros("compare_pd")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('distances', type=pathlib.Path, nargs='+')

    args = parser.parse_args()

    names = make_useful_names([d.name for d in args.distances])

    for space in ['rope', 'robot', 'env']:
        print(Fore.CYAN + space + Fore.RESET)
        space_idx = space_to_idx(space)

        distances_matrices = []
        plausibilities = []
        diversities = []
        for distances_file in args.distances:
            aug_examples_matrix, data_examples_matrix, distances_matrix = format_distances(results_dir=distances_file,
                                                                                           space_idx=space_idx)
            distances_matrices.append(distances_matrix)
            diversities.append(compute_diversity(distances_matrix, aug_examples_matrix, data_examples_matrix))
            plausibilities.append(compute_plausibility(distances_matrix, aug_examples_matrix, data_examples_matrix))

        print(Fore.GREEN + 'Diversity' + Fore.RESET)
        diversities_dict = {}
        for (name, d) in zip(names, diversities):
            diversities_dict[name] = d
            print(f'{name:.30s}  mean: {np.mean(d):.3f}, std: {np.std(d):.3f}')
        print()
        print(dict_to_pvalue_table(diversities_dict))
        print()
        print()

        print(Fore.GREEN + 'Plausibility' + Fore.RESET)
        plausibilities_dict = {}
        for (name, d) in zip(names, plausibilities):
            plausibilities_dict[name] = d
            print(f'{name:.30s}  mean: {np.mean(d):.3f}, std: {np.std(d):.3f}')
        print(dict_to_pvalue_table(plausibilities_dict))
        print()


if __name__ == '__main__':
    main()
