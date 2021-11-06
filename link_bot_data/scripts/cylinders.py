#!/usr/bin/env python
import argparse
import pathlib
import queue
from multiprocessing import Process, Queue

import numpy as np
from tqdm import tqdm

from link_bot_data.merge_pkls import merge_pkls


def _collect_dynamics_data(i, name, n_trajs_per, params, q):
    import sys
    from link_bot_data.base_collect_dynamics_data import collect_dynamics_data
    from arc_utilities.ros_init import RosContext
    sys.stdout = open(f'.log_{i}', 'w')
    sys.stderr = sys.stdout
    with RosContext(f'collect_dynamics_data_{i}'):
        for dataset_dir, n_trajs_per in collect_dynamics_data(collect_dynamics_params=params,
                                                              seed=i,
                                                              verbose=0,
                                                              n_trajs=n_trajs_per,
                                                              nickname=f'{name}-{i}'):
            q.put((dataset_dir, n_trajs_per))


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

    trajs_splits = np.array_split(range(args.n_trajs_total), args.j)
    pqs = []
    for i, trajs_split in enumerate(trajs_splits):
        q = Queue()
        p = Process(target=_collect_dynamics_data, args=(i, args.name, len(trajs_split), args.params, q))
        pqs.append((p, q))
        p.start()

    dataset_dirs = []
    for dataset_dir, n_trajs_done in tqdm(generate(pqs), total=args.n_trajs_total):
        if dataset_dir is not None:
            dataset_dirs.append(dataset_dir)

    outdir = pathlib.Path('fwd_model_data') / args.name
    merge_pkls(outdir, dataset_dirs, quiet=True)
    print(outdir)


if __name__ == '__main__':
    main()
