#!/usr/bin/env python
import argparse
import pathlib
from time import perf_counter

from progressbar import progressbar

from arc_utilities import ros_init
from link_bot_data.load_dataset import guess_load_dataset
from link_bot_data.progressbar_widgets import mywidgets
from moonshine.simple_profiler import SimpleProfiler


@ros_init.with_ros("test_dataset_perf")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('--mode', choices=['train', 'test', 'val'], default='train')
    parser.add_argument('--n-repetitions', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)

    args = parser.parse_args()

    dataset_loader = guess_load_dataset(args.dataset_dir)

    dataset = dataset_loader.get_datasets(mode=args.mode)
    dataset = dataset.batch(args.batch_size)

    p = SimpleProfiler()

    p.start()
    t0 = perf_counter()
    for i in range(args.n_repetitions):
        for _ in progressbar(dataset, widgets=mywidgets):
            p.lap()
            dt = perf_counter() - t0
            if dt > 30:
                break
    print(p)


if __name__ == '__main__':
    main()
