#!/usr/bin/env python
import argparse
import pathlib
import queue
import traceback
from copy import deepcopy
from multiprocessing import Process, Queue

import numpy as np
from colorama import Fore
from tqdm import tqdm

from arc_utilities.path_utils import rm_tree
from dm_envs.add_velocity_to_dynamics_dataset import add_velocity_to_dataset
from link_bot_data.merge_pkls import merge_pkls


def _collect_dynamics_data(i, name, n_trajs_per, params, q):
    import sys
    from link_bot_data.base_collect_dynamics_data import collect_dynamics_data
    from arc_utilities.ros_init import RosContext
    # sys.stdout = open(f'.log_{i}', 'w')
    # sys.stderr = sys.stdout
    with RosContext(f'collect_dynamics_data_{i}'):
        try:
            for dataset_dir, n_trajs_per in collect_dynamics_data(collect_dynamics_params=params,
                                                                  seed=i,
                                                                  verbose=0,
                                                                  n_trajs=n_trajs_per,
                                                                  nickname=f'{name}-{i}'):
                q.put((dataset_dir, n_trajs_per))
        except Exception as e:
            sys.stderr = sys.__stderr__
            sys.stdout = sys.__stdout__
            traceback.print_exc()
            print(e)


def generate(pqs):
    while True:
        all_done = True
        num_trajs_collected_total = 0
        for p, q in pqs:
            try:
                dataset_dir, num_trajs_collected = q.get()
                num_trajs_collected_total += num_trajs_collected
                if dataset_dir is None:
                    all_done = False
                    yield None, num_trajs_collected_total
                else:
                    yield dataset_dir, num_trajs_collected_total
            except queue.Empty:
                pass
        if all_done:
            return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('j', type=int, help='number of processes')
    parser.add_argument('n_trajs_total', type=int, help='number of trajs total')
    parser.add_argument('name', help='name of output dir')
    parser.add_argument("params", type=pathlib.Path, default=pathlib.Path('collect_dynamics_params/cylinders.hjson'))

    args = parser.parse_args()

    dataset_dirs = collect_data_in_parallel(args.j, args.n_trajs_total, args.name, args.params)

    outdir = pathlib.Path('fwd_model_data') / args.name
    merge_pkls(outdir, dataset_dirs, quiet=True)
    # for dataset_dir in dataset_dirs:
    #     rm_tree(dataset_dir)
    outdir = add_velocity_to_dataset(outdir)
    print(Fore.GREEN + outdir.as_posix() + Fore.RESET)


def collect_data_in_parallel(j, n_trajs_total, name, params):
    trajs_splits = np.array_split(range(n_trajs_total), j)
    pqs = []
    for i, trajs_split in enumerate(trajs_splits):
        q = Queue()
        p = Process(target=_collect_dynamics_data, args=(i, name, len(trajs_split), params, q))
        pqs.append((p, q))
        p.start()
    dataset_dirs = []
    for dataset_dir, n_trajs_done in tqdm(generate(pqs), total=n_trajs_total):
        if dataset_dir is not None:
            dataset_dirs.append(dataset_dir)
    return dataset_dirs


if __name__ == '__main__':
    main()
