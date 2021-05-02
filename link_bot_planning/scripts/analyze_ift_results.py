#!/usr/bin/env python
import argparse
import pathlib
import pickle

import colorama
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from colorama import Fore

import rospy
from arc_utilities.filesystem_utils import get_all_subdirs
from link_bot_planning.analysis.figspec import FigSpec, get_data_for_figure
from link_bot_planning.analysis.results_figures import *
from link_bot_planning.analysis.results_metrics import *
from link_bot_planning.analysis.results_utils import load_order, add_number_to_method_name
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.pycommon import quote_string
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

    metrics_funcs = [
        num_steps,
        task_error,
        any_solved,
        success,
        normalized_model_error,
    ]
    metrics_names = [func.__name__ for func in metrics_funcs]
    column_names = metrics_names

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
            metrics = pickle.load(pickle_file)
    else:
        rospy.loginfo(Fore.GREEN + f"Generating metrics")

        data = []
        index_tuples = []
        for method_name, (dirs, _) in results_dirs_dict.items():
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
                    index_tuples.append([method_name, iteration, file_idx])
                    data.append([metric_func(scenario, metadata, datum) for metric_func in metrics_funcs])

        index = pd.MultiIndex.from_tuples(index_tuples, names=["method_name", "iteration_idx", "file_idx"])
        metrics = pd.DataFrame(data=data, index=index, columns=column_names)

        with pickle_filename.open("wb") as pickle_file:
            pickle.dump(metrics, pickle_file)
        rospy.loginfo(Fore.GREEN + f"Pickling metrics to {pickle_filename}")

    # Figures & Tables
    title = eval(quote_string(analysis_params['experiment_name']))
    figspecs = [
        FigSpec(fig=LinePlot(analysis_params, xlabel="Num Steps", ylabel='Normalized Model Error'),
                reductions={num_steps.__name__:  [None, 'cumsum', 'sum'],
                            normalized_model_error.__name__: [None, None, 'mean']}),
        FigSpec(fig=LinePlot(analysis_params, xlabel="Num Steps", ylabel='Task Error'),
                reductions={num_steps.__name__:  [None, 'cumsum', 'sum'],
                            task_error.__name__: [None, None, 'mean']}),
        FigSpec(fig=LinePlot(analysis_params, xlabel="Num Steps", ylabel='Solved'),
                reductions={num_steps.__name__:  [None, 'cumsum', 'sum'],
                            any_solved.__name__: [None, None, 'mean']}),
        FigSpec(fig=LinePlot(analysis_params, xlabel="Num Steps", ylabel='Percent Success'),
                reductions={num_steps.__name__:  [None, 'cumsum', 'sum'],
                            success.__name__: [None, None, 'mean']}),
        FigSpec(fig=LinePlot(analysis_params, xlabel="Iterations", ylabel='Task Error'),
                reductions={task_error.__name__: [None, None, 'mean']},
                axis_names=['y']),
        FigSpec(fig=LinePlot(analysis_params, xlabel="Iterations", ylabel='Solved'),
                reductions={any_solved.__name__: [None, None, 'mean']},
                axis_names=['y']),
    ]

    for spec in figspecs:
        data_for_figure = get_data_for_figure(spec, metrics)

        spec.fig.make_figure(data_for_figure, method_names)
        spec.fig.save_figure(out_dir)
        spec.fig.fig.suptitle(title)

    # make_tables(tables, analysis_params, sort_order_dict, table_format, tables_filename)

    if not args.no_plot:
        for spec in figspecs:
            spec.fig.fig.set_tight_layout(True)
        plt.show()


def main():
    colorama.init(autoreset=True)

    rospy.init_node("analyse_ift_results")
    np.set_printoptions(suppress=True, precision=4, linewidth=220)

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
