#!/usr/bin/env python
import argparse
import csv
import pathlib

import colorama
import numpy as np
import tabulate

import rospy
from analysis import results_utils, results_metrics
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.get_scenario import get_scenario
from moonshine.filepath_tools import load_json_or_hjson


def metrics_main(args):
    metadata = load_json_or_hjson(args.results_dir, 'metadata')
    scenario = get_scenario(metadata['scenario'])

    metrics_funcs = [
        results_metrics.task_error,
        results_metrics.num_recovery_actions,
        results_metrics.recovery_success,
        results_metrics.total_time,
        results_metrics.planning_time,
    ]

    rows = []
    for trial_idx, datum in results_utils.trials_generator(args.results_dir):
        row = [trial_idx] + [f(scenario, metadata, datum) for f in metrics_funcs]
        rows.append(row)

    rows = sorted(rows)

    headers = ['trial idx'] + [f.__name__ for f in metrics_funcs]
    print(tabulate.tabulate(rows, headers=headers, tablefmt=tabulate.simple_separated_format("\t"), numalign='left'))

    with open(args.results_dir / 'results_stats.txt', 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(headers)
        writer.writerows(rows)


def main():
    colorama.init(autoreset=True)

    rospy.init_node("print_results_stats")
    np.set_printoptions(suppress=True, precision=4, linewidth=180)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('results_dir', help='results directory', type=pathlib.Path)

    args = parser.parse_args()

    metrics_main(args)


if __name__ == '__main__':
    main()
