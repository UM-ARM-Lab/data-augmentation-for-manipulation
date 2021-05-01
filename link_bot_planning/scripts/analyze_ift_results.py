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
from link_bot_planning.analysis.figspec import FigSpec, reduce_metrics_for_figure
from link_bot_planning.analysis.results_figures import *
from link_bot_planning.analysis.results_metrics import *
from link_bot_planning.analysis.results_metrics import MeanReduction, NoReduction, SumReduction, CumSumReduction
from link_bot_planning.analysis.results_utils import load_order, add_number_to_method_name
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.serialization import load_gzipped_pickle
from moonshine.filepath_tools import load_hjson, load_json_or_hjson
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

    metric_funcs = [
        num_steps,
        task_error,
    ]

    results_dirs_ordered = load_order(prompt_order=args.order, directories=planning_results_dirs, out_dir=out_dir)

    results_dirs_dict = {}
    sort_order_dict = {}
    for idx, results_dir in enumerate(results_dirs_ordered):
        log = load_hjson(results_dir.parent / 'logfile.hjson')
        from_env = log['from_env']
        to_env = log['to_env']
        if args.use_nickname:
            method_name = log['nickname']
        else:
            method_name = f'{from_env}_to_{to_env}'
        subfolders = sorted(get_all_subdirs([results_dir]))
        while method_name in results_dirs_dict:
            method_name = add_number_to_method_name(method_name)
        results_dirs_dict[method_name] = (subfolders, log)
        sort_order_dict[method_name] = idx
    method_names = list(sort_order_dict.keys())

    tables_filename = out_dir / 'tables.txt'
    with tables_filename.open("w") as tables_file:
        tables_file.truncate()

    pickle_filename = out_dir / f"metrics.pkl"
    if pickle_filename.exists() and not args.regenerate:
        rospy.loginfo(Fore.GREEN + f"Loading existing metrics from {pickle_filename}")
        with pickle_filename.open("rb") as pickle_file:
            metrics_indices, metrics = pickle.load(pickle_file)
    else:
        rospy.loginfo(Fore.GREEN + f"Generating metrics")

        metrics: Dict = {}
        metrics_indices: Dict = {}
        for metric_func in metric_funcs:
            metrics[metric_func.__name__] = []
            metrics_indices[metric_func.__name__] = []

        for method_idx, (method_name, (dirs, _)) in enumerate(results_dirs_dict.items()):
            print(Fore.GREEN + f"processing {method_name} {[d.name for d in dirs]}")

            logfile = load_json_or_hjson(dirs[0].parent.parent, 'logfile')
            scenario = get_scenario(logfile['planner_params']['scenario'])

            for iteration, iteration_folder in enumerate(dirs):
                assert str(iteration) in iteration_folder.name  # sanity check

                metadata = load_json_or_hjson(iteration_folder, 'metadata')

                # NOTE: even though this is slow, parallelizing is not easy because "scenario" cannot be pickled
                metrics_filenames = list(iteration_folder.glob("*_metrics.pkl.gz"))
                for file_idx, metrics_filename in enumerate(metrics_filenames):
                    datum = load_gzipped_pickle(metrics_filename)
                    for metric_func in metric_funcs:
                        metric_value = metric_func(scenario, metadata, datum)
                        indices = [method_idx, iteration, file_idx]
                        metrics_indices[metric_func.__name__].append(indices)
                        metrics[metric_func.__name__].append(metric_value)

        with pickle_filename.open("wb") as pickle_file:
            pickle.dump((metrics_indices, metrics), pickle_file)
        rospy.loginfo(Fore.GREEN + f"Pickling metrics to {pickle_filename}")

    # Figures & Tables
    figures = [
        FigSpec(fig=LinePlot(analysis_params, ylabel='Task Error'),
                metrics_indices=[metrics_indices[num_steps.__name__],
                                 metrics_indices[task_error.__name__]],
                metrics=[metrics[num_steps.__name__],
                         metrics[task_error.__name__]],
                reductions=[[NoReduction(), CumSumReduction(), SumReduction()],
                            [NoReduction(), NoReduction(), MeanReduction()]]),
        # LinePlotAcrossSteps(analysis_params, metrics[TaskError], 'Task Error'),
        # LinePlotAcrossIterations(analysis_params, metrics[TotalTime], 'Total Time'),
        # LinePlotAcrossIterations(analysis_params, metrics[PlanningTime], 'Planning Time'),
        # LinePlotAcrossIterations(analysis_params, metrics[NormalizedModelError], 'Normalized Model Error'),
        # LinePlotAcrossIterations(analysis_params, metrics[Successes], 'Percentage Success'),
        # LinePlotAcrossIterations(analysis_params, metrics[PlannerSolved], 'Planner Returned Solved'),
        # MovingAverageAcrossItersLinePlot(analysis_params, metrics[NormalizedModelError],
        #                                  'Normalized Model Error (moving average)'),
        # MovingAverageAcrossItersLinePlot(analysis_params, metrics[Successes], 'Percentage Success (moving average)'),
        # CumulativeLinePlotAcrossIters(analysis_params, metrics[NormalizedModelError],
        #                               'Normalized Model Error (cumulative)'),
        # CumulativeLinePlotAcrossIters(analysis_params, metrics[Successes], 'Percentage Success (cumulative)'),
    ]

    for figspec in figures:
        data = reduce_metrics_for_figure(figspec)

        figure = figspec.fig
        figure.params = analysis_params
        # figure.sort_methods(sort_order_dict)
        # figure.enumerate_methods()
        figure.make_figure(data, method_names)
        figure.save_figure(out_dir)

    # make_figures(figures, analysis_params, sort_order_dict, out_dir)
    # make_tables(tables, analysis_params, sort_order_dict, table_format, tables_filename)

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
    parser.add_argument('--use-nickname', action='store_true')
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
