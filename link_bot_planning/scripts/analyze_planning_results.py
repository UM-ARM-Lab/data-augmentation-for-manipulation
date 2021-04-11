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
from link_bot_planning.analysis.results_metrics import load_analysis_params, generate_per_trial_metrics
from link_bot_planning.analysis.results_utils import save_order, load_sort_order, load_order
from link_bot_pycommon.args import my_formatter
from moonshine.filepath_tools import load_json_or_hjson
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(0.1)


def metrics_main(args):
    analysis_params = load_analysis_params(args.analysis_params)

    # The default for where we write results
    out_dir = args.results_dirs[0]
    print(f"Writing analysis to {out_dir}")

    unique_comparison_name = "-".join([p.name for p in args.results_dirs])

    subfolders = get_all_subdirs(args.results_dirs)

    if args.latex:
        table_format = 'latex_raw'
    else:
        table_format = 'fancy_grid'

    subfolders_ordered = load_order(prompt_order=args.order, directories=subfolders, out_dir=out_dir)

    tables_filename = out_dir / 'tables.txt'
    with tables_filename.open("w") as tables_file:
        tables_file.truncate()

    sort_order_dict = {}
    method_names = []
    for sort_idx, subfolder in enumerate(subfolders_ordered):
        metadata = load_json_or_hjson(subfolder, 'metadata')
        method_name = metadata['planner_params'].get('method_name', subfolder.name)
        sort_order_dict[method_name] = sort_idx
        method_names.append(method_name)

    pickle_filename = out_dir / f"{unique_comparison_name}-metrics.pkl"
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
        metrics = generate_per_trial_metrics(analysis_params, subfolders_ordered, method_names)

        with pickle_filename.open("wb") as pickle_file:
            pickle.dump(metrics, pickle_file)
        rospy.loginfo(Fore.GREEN + f"Pickling metrics to {pickle_filename}")

    figures = [
        TaskErrorLineFigure(analysis_params, metrics[TaskError]),
        BarChartPercentagePerMethodFigure(analysis_params, metrics[Successes], '% Success'),
        violin_plot(analysis_params, metrics[TaskError], 'Task Error'),
        box_plot(analysis_params, metrics[NRecoveryActions], "Num Recovery Actions"),
        box_plot(analysis_params, metrics[TotalTime], 'Total Time'),
        violin_plot(analysis_params, metrics[TotalTime], 'Total Time'),
        box_plot(analysis_params, metrics[NPlanningAttempts], 'Num Planning Attempts'),
        box_plot(analysis_params, metrics[NMERViolations], 'Num MER Violations'),
        box_plot(analysis_params, metrics[NormalizedModelError], 'Normalized Model Error'),
        box_plot(analysis_params, metrics[PlanningTime], 'Planning Time'),
        box_plot(analysis_params, metrics[PercentageMERViolations], '% MER Violations'),
        BarChartPercentagePerMethodFigure(analysis_params, metrics[PlannerSolved], '% Planner Returned Solved'),
    ]

    make_figures(figures, analysis_params, sort_order_dict, table_format, tables_filename, out_dir)

    if not args.no_plot:
        for figure in figures:
            figure.fig.set_tight_layout(True)
        plt.show()


def main():
    colorama.init(autoreset=True)

    rospy.init_node("analyse_planning_results")
    np.set_printoptions(suppress=True, precision=4, linewidth=180)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('results_dirs', help='results directory', type=pathlib.Path, nargs='+')
    parser.add_argument('analysis_params', type=pathlib.Path)
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
