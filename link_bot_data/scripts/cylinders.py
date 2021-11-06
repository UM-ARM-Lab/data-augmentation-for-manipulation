#!/usr/bin/env python
import argparse
import pathlib
from multiprocessing import Process, Queue
import queue

import numpy as np

from arc_utilities.ros_init import RosContext
from link_bot_data.base_collect_dynamics_data import collect_dynamics_data
from link_bot_data.merge_pkls import merge_pkls


def _collect_dynamics_data(i, name, n_trajs_per, params, q):
    import sys
    # sys.stdout = open(f'.log_{i}', 'w')
    # sys.stderr = sys.stdout
    with RosContext(f'collect_dynamics_data_{i}'):
        for done, dataset_dir, n_trajs_per in collect_dynamics_data(collect_dynamics_params=params,
                                                                    seed=i,
                                                                    verbose=0,
                                                                    n_trajs=n_trajs_per,
                                                                    nickname=f'{name}-{i}'):
            q.put((done, dataset_dir, n_trajs_per))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('j', type=int, help='number of processes')
    parser.add_argument('n_trajs_total', type=int, help='number of trajs total')
    parser.add_argument('name', help='name of output dir')
    parser.add_argument("params", type=pathlib.Path, default=pathlib.Path('collect_dynamics_params/cylinders.hjson'))

    args = parser.parse_args()

    trajs_splits = np.array_split(range(args.n_trajs_total), args.j)
    pqs = []
    for i, trajs_split in enumerate(trajs_splits):
        q = Queue()
        p = Process(target=_collect_dynamics_data, args=(i, args.name, len(trajs_split), args.params, q))
        pqs.append((p,q))
        p.start()
    print("Started")

    dataset_dirs = []
    while True:
        all_done = True
        num_trajs_collected_total = 0
        for p, q in pqs:
            try:
                done, dataset_dir, num_trajs_collected = q.get()
                num_trajs_collected_total += num_trajs_collected
                if not done:
                    all_done = False
                else:
                    dataset_dirs.append(dataset_dir)
            except queue.Empty:
                pass
        print(num_trajs_collected_total)
        if all_done:
            break
    print("done!")

    outdir = pathlib.Path('fwd_model_data') / args.name
    merge_pkls(outdir, dataset_dirs, quiet=True)
    print(outdir)


if __name__ == '__main__':
    main()
