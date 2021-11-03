#!/usr/bin/env python
import argparse
import pathlib
from multiprocessing import Pool

import numpy as np

from arc_utilities.ros_init import RosContext
from link_bot_data.base_collect_dynamics_data import collect_dynamics_data
from link_bot_data.merge_pkls import merge_pkls


def _collect_dynamics_data(args):
    import sys
    i, name, n_trajs_per = args
    sys.stderr = open(f'.log_{i}', 'w')
    # sys.stdout = sys.stdout
    with RosContext(f'collect_dynamics_data_{i}'):
        dataset_dir = collect_dynamics_data(
            collect_dynamics_params=pathlib.Path('collect_dynamics_params/cylinders.hjson'),
            seed=i,
            verbose=0,
            n_trajs=n_trajs_per,
            nickname=f'{name}-{i}')
    return dataset_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('j', type=int, help='number of processes')
    parser.add_argument('n_trajs_total', type=int, help='number of trajs total')
    parser.add_argument('name', help='name of output dir')

    args = parser.parse_args()

    with Pool(args.j) as p:
        trajs_splits = np.array_split(range(args.n_trajs_total), args.j)
        process_args = [(i, args.name, len(trajs_split)) for i, trajs_split in enumerate(trajs_splits)]
        dataset_dirs = list(p.imap_unordered(_collect_dynamics_data, process_args))
    outdir = pathlib.Path('fwd_model_data') / args.name
    merge_pkls(outdir, dataset_dirs, quiet=True)
    print(outdir)


if __name__ == '__main__':
    main()
