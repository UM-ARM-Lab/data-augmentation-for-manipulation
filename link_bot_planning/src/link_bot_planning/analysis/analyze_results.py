import pickle
from typing import Callable

import pandas as pd
from colorama import Fore

# noinspection PyUnresolvedReferences
from link_bot_planning.analysis.results_figures import *

import rospy
from arc_utilities.filesystem_utils import get_all_subdirs
from link_bot_planning.analysis.figspec import DEFAULT_AXES_NAMES, FigSpec
from link_bot_planning.analysis.results_metrics import num_trials, num_steps, task_error, any_solved, success, \
    normalized_model_error, num_recovery_actions
from link_bot_planning.analysis.results_utils import load_order, add_number_to_method_name
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.serialization import load_gzipped_pickle
from moonshine.filepath_tools import load_hjson, load_json_or_hjson


def get_metrics2(args, out_dir, planning_results_dirs, get_method_name: Callable, get_metadata: Callable):
    metrics_funcs = [
        num_trials,
        num_steps,
        task_error,
        any_solved,
        success,
        normalized_model_error,
        num_recovery_actions,
    ]
    metrics_names = [func.__name__ for func in metrics_funcs]
    column_names = metrics_names
    results_dirs_ordered = load_order(prompt_order=args.order, directories=planning_results_dirs, out_dir=out_dir)

    with (out_dir / 'info.txt').open('w') as info_file:
        for f in planning_results_dirs:
            info_file.write(f.as_posix() + '\n')

    results_dirs_dict = {}
    sort_order_dict = {}
    for idx, results_dir in enumerate(results_dirs_ordered):
        method_name = get_method_name(results_dir)
        results_dirs_dict[method_name] = results_dir
        print(method_name, results_dir)
        sort_order_dict[method_name] = idx

    method_names = list(sort_order_dict.keys())

    tables_filename = out_dir / 'tables.txt'
    with tables_filename.open("w") as tables_file:
        tables_file.truncate()

    pickle_filename = out_dir / f"metrics.pkl"
    if pickle_filename.exists() and not args.regenerate:
        rospy.loginfo(Fore.GREEN + f"Loading existing metrics from {pickle_filename}" + Fore.RESET)
        with pickle_filename.open("rb") as pickle_file:
            metrics = pickle.load(pickle_file)
    else:
        rospy.loginfo(Fore.GREEN + f"Generating metrics" + Fore.RESET)

        data = []
        index_tuples = []
        for method_name, results_dir in results_dirs_dict.items():
            metadata = get_metadata(results_dir)
            scenario = get_scenario(metadata['planner_params']['scenario'])

            metadata = load_json_or_hjson(results_dir, 'metadata')

            # NOTE: even though this is slow, parallelizing is not easy because "scenario" cannot be pickled
            metrics_filenames = list(results_dir.glob("*_metrics.pkl.gz"))
            for file_idx, metrics_filename in enumerate(metrics_filenames):
                datum = load_gzipped_pickle(metrics_filename)
                index_tuples.append([method_name, file_idx])
                data.append([metric_func(scenario, metadata, datum) for metric_func in metrics_funcs])

        index = pd.MultiIndex.from_tuples(index_tuples, names=["method_name", "file_idx"])
        metrics = pd.DataFrame(data=data, index=index, columns=column_names)

        with pickle_filename.open("wb") as pickle_file:
            pickle.dump(metrics, pickle_file)
        rospy.loginfo(Fore.GREEN + f"Pickling metrics to {pickle_filename}" + Fore.RESET)

    return method_names, metrics
def get_metrics(args, out_dir, planning_results_dirs, get_method_name: Callable, get_metadata: Callable):
    metrics_funcs = [
        num_trials,
        num_steps,
        task_error,
        any_solved,
        success,
        normalized_model_error,
        num_recovery_actions,
    ]
    metrics_names = [func.__name__ for func in metrics_funcs]
    column_names = metrics_names
    results_dirs_ordered = load_order(prompt_order=args.order, directories=planning_results_dirs, out_dir=out_dir)

    with (out_dir / 'info.txt').open('w') as info_file:
        for f in planning_results_dirs:
            info_file.write(f.as_posix() + '\n')

    results_dirs_dict = {}
    sort_order_dict = {}
    for idx, results_dir in enumerate(results_dirs_ordered):
        method_name = get_method_name(results_dir)
        subfolders = sorted(get_all_subdirs([results_dir]))
        while method_name in results_dirs_dict:
            method_name = add_number_to_method_name(method_name)
        results_dirs_dict[method_name] = subfolders
        sort_order_dict[method_name] = idx

    method_names = list(sort_order_dict.keys())

    tables_filename = out_dir / 'tables.txt'
    with tables_filename.open("w") as tables_file:
        tables_file.truncate()

    pickle_filename = out_dir / f"metrics.pkl"
    if pickle_filename.exists() and not args.regenerate:
        rospy.loginfo(Fore.GREEN + f"Loading existing metrics from {pickle_filename}" + Fore.RESET)
        with pickle_filename.open("rb") as pickle_file:
            metrics = pickle.load(pickle_file)
    else:
        rospy.loginfo(Fore.GREEN + f"Generating metrics" + Fore.RESET)

        data = []
        index_tuples = []
        for method_name, results_dir in results_dirs_dict.items():
            print(Fore.GREEN + f"processing {method_name} {[d.name for d in results_dir]}" + Fore.RESET)

            for iteration, iteration_folder in enumerate(results_dir):
                assert str(iteration) in iteration_folder.name  # sanity check

                metadata = get_metadata(iteration_folder)
                scenario = get_scenario(metadata['planner_params']['scenario'])

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
        rospy.loginfo(Fore.GREEN + f"Pickling metrics to {pickle_filename}" + Fore.RESET)

    return method_names, metrics


def load_figspecs(analysis_params, args):
    figures_config = load_hjson(args.figures_config)
    figspecs = []
    for fig_config in figures_config:
        figure_type = eval(fig_config.pop('type'))
        reductions = fig_config.pop('reductions')
        if 'axes_names' in fig_config:
            axes_names = fig_config.pop('axes_names')
        else:
            axes_names = DEFAULT_AXES_NAMES

        fig = figure_type(analysis_params, **fig_config)

        figspec = FigSpec(fig=fig, reductions=reductions, axes_names=axes_names)
        figspecs.append(figspec)
    return figspecs
