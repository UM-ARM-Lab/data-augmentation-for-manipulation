#!/usr/bin/env python
import argparse
import pathlib
import pickle
from typing import Dict

import colorama
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore

import rospy
from arc_utilities.filesystem_utils import get_all_subdirs
from link_bot_planning.analysis.results_figures import *
from link_bot_planning.analysis.results_figures import make_figures
from link_bot_planning.analysis.results_metrics import *
from link_bot_planning.analysis.results_metrics import generate_multi_trial_metrics
from link_bot_planning.analysis.results_utils import load_order
from link_bot_pycommon.args import my_formatter
from moonshine.filepath_tools import load_hjson
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(0.1)


def metrics_main(args):
    analysis_params = load_analysis_params(args.analysis_params)

    # The default for where we write results
    planning_results_dirs = [d / 'planning_results' for d in args.ift_dirs]
    out_dir = args.ift_dirs[0]
    print(f"Writing analysis to {out_dir}")

    if args.latex:
        table_format = 'latex_raw'
    else:
        table_format = 'fancy_grid'

    results_dirs_ordered = load_order(prompt_order=args.order, directories=planning_results_dirs, out_dir=out_dir)

    results_dirs_dict = {}
    sort_order_dict = {}
    for idx, results_dir in enumerate(results_dirs_ordered):
        log = load_hjson(results_dir.parent / 'logfile.hjson')
        from_env = log['from_env']
        to_env = log['to_env']
        method_name = f'{from_env}_to_{to_env}'
        subfolders = sorted(get_all_subdirs([results_dir]))
        if method_name in results_dirs_dict:
            method_name = method_name + '2'
        results_dirs_dict[method_name] = (subfolders, log)
        sort_order_dict[method_name] = idx

    tables_filename = out_dir / 'tables.txt'
    with tables_filename.open("w") as tables_file:
        tables_file.truncate()

    pickle_filename = out_dir / f"metrics.pkl"
    if pickle_filename.exists() and not args.regenerate:
        rospy.loginfo(Fore.GREEN + f"Loading existing metrics from {pickle_filename}")
        with pickle_filename.open("rb") as pickle_file:
            metrics: Dict[type, TrialMetrics] = pickle.load(pickle_file)

        # update the analysis params so we don't need to regenerate metrics
        for metric in metrics:
            metric.params = analysis_params

        with pickle_filename.open("wb") as pickle_file:
            pickle.dump(metrics, pickle_file)
        rospy.loginfo(Fore.GREEN + f"Pickling metrics to {pickle_filename}")
    else:
        rospy.loginfo(Fore.GREEN + f"Generating metrics")
        metrics = generate_multi_trial_metrics(analysis_params, results_dirs_dict)

        with pickle_filename.open("wb") as pickle_file:
            pickle.dump(metrics, pickle_file)
        rospy.loginfo(Fore.GREEN + f"Pickling metrics to {pickle_filename}")

    figures = [
        LinePlotAcrossIterationsFigure(analysis_params, metrics[TaskError], 'Task Error'),
        LinePlotAcrossIterationsFigure(analysis_params, metrics[TotalTime], 'Total Time'),
        LinePlotAcrossIterationsFigure(analysis_params, metrics[PlanningTime], 'Planning Time'),
        LinePlotAcrossIterationsFigure(analysis_params, metrics[NormalizedModelError], 'Normalized Model Error'),
        LinePlotAcrossIterationsFigure(analysis_params, metrics[Successes], 'Percentage Success'),
        LinePlotAcrossIterationsFigure(analysis_params, metrics[PlannerSolved], 'Planner Returned Solved'),
        MWALinePlotAcrossIterationsFigure(analysis_params, metrics[NormalizedModelError],
                                          'Normalized Model Error (moving average)'),
        MWALinePlotAcrossIterationsFigure(analysis_params, metrics[Successes], 'Percentage Success (moving average)'),
    ]

    make_figures(figures, analysis_params, sort_order_dict, table_format, tables_filename, out_dir)

    if not args.no_plot:
        for figure in figures:
            figure.fig.set_tight_layout(True)
        plt.show()


def main():
    colorama.init(autoreset=True)

    rospy.init_node("analyse_ift_results")
    np.set_printoptions(suppress=True, precision=4, linewidth=180)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('ift_dirs', help='results directory', type=pathlib.Path, nargs='+')
    parser.add_argument('--analysis-params', type=pathlib.Path,
                        default=pathlib.Path("analysis_params/env_across_methods.json"))
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--show-all-trials', action='store_true')
    parser.add_argument('--latex', action='store_true')
    parser.add_argument('--order', action='store_true')
    parser.add_argument('--regenerate', action='store_true')
    parser.add_argument('--debug', action='store_true', help='will only run on a few examples to speed up debugging')
    parser.add_argument('--style', default='slides')
    parser.set_defaults(func=metrics_main)

    args = parser.parse_args()

    plt.style.use(args.style)

    metrics_main(args)


if __name__ == '__main__':
    main()
