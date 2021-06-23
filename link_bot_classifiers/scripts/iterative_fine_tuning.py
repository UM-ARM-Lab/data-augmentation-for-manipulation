#!/usr/bin/env python
import argparse
import logging
import pathlib

import pandas as pd
import tensorflow as tf

from arc_utilities import ros_init
from link_bot_classifiers.iterative_fine_tuning import iterative_fine_tuning
from link_bot_pycommon.args import run_subparsers
from moonshine.filepath_tools import load_hjson


@ros_init.with_ros("iterative_fine_tuning_run")
def run_main(args):
    iterative_fine_tuning(**vars(args))


@ros_init.with_ros("iterative_fine_tuning_plot")
def plot_main(args):
    data = []
    # the items in 'row' must match this
    column_names = [
        'augmentation_type',
        'checkpoint',
        'iter_idx',
        'shuffle_seed',
        'dataset_dir',
        'metric_name',
        'metric_value',
    ]

    for logdir in args.logdirs:
        results = load_hjson(logdir / 'logfile.hjson')
        for k, v in results.items():
            iter_idx = int(k)
            results_i = v['results']
            augmentation_type = results_i['augmentation_type']
            seed = results_i['seed']
            checkpoint = results_i['checkpoint']
            results_results = results_i['results']
            for result in results_results:
                row = [
                    augmentation_type,
                    checkpoint,
                    iter_idx,
                    seed,
                    result['info']['dataset_dir'],
                    result['metric_name'],
                    result['metric_value'],
                ]
                data.append(row)

    df = pd.DataFrame(data, columns=column_names)
    print(df)


def main():
    tf.get_logger().setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    run_parser = subparsers.add_parser('run')
    run_parser.add_argument('training_dataset_dir', type=pathlib.Path)
    run_parser.add_argument('checkpoint', type=pathlib.Path)
    run_parser.add_argument('proxy_datasets_info', type=pathlib.Path,
                            help='a hjson file listing dataset and metric names')
    run_parser.add_argument('seed', type=int)
    run_parser.add_argument('nickname')
    run_parser.add_argument('--params', '-p', type=pathlib.Path, help='an hjson file to override the model hparams')
    run_parser.add_argument('--augmentation-config-dir', type=pathlib.Path, help='dir of pkl files with state/env')
    run_parser.add_argument('--batch-size', type=int, default=24)
    run_parser.add_argument('--epochs', type=int, default=50)
    run_parser.set_defaults(func=run_main)

    plot_parser = subparsers.add_parser('plot')
    plot_parser.add_argument('logdirs', type=pathlib.Path, nargs='+')
    plot_parser.set_defaults(func=plot_main)

    run_subparsers(parser)


if __name__ == '__main__':
    main()
