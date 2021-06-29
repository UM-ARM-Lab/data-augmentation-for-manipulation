import pathlib
import pickle
from typing import Callable, List

import pandas as pd
from colorama import Fore
from halo import Halo
from progressbar import progressbar

import link_bot_planning.analysis.results_metrics
import rospy
from arc_utilities.filesystem_utils import get_all_subdirs
from link_bot_data.progressbar_widgets import mywidgets
from link_bot_planning.analysis.figspec import DEFAULT_AXES_NAMES, FigSpec, TableSpec
from link_bot_planning.analysis.results_metrics import metrics_funcs
from link_bot_planning.analysis.results_metrics import metrics_names
# noinspection PyUnresolvedReferences
from link_bot_planning.analysis.results_tables import *
from link_bot_planning.analysis.results_utils import load_order, add_number_to_method_name
from link_bot_pycommon.get_scenario import get_scenario, get_scenario_cached
from link_bot_pycommon.pandas_utils import df_append
from link_bot_pycommon.serialization import load_gzipped_pickle
from moonshine.filepath_tools import load_hjson, load_json_or_hjson

column_names = [
                   'method_name',
                   'seed',
                   'ift_iteration',
                   'trial_idx',
                   'uuid',
               ] + metrics_names


def get_metrics2(args, out_dir, planning_results_dirs, get_method_name: Callable, get_metadata: Callable):
    results_dirs_ordered = load_order(prompt_order=args.order, directories=planning_results_dirs, out_dir=out_dir)

    with (out_dir / 'info.txt').open('w') as info_file:
        for f in planning_results_dirs:
            info_file.write(f.as_posix() + '\n')

    results_dirs_dict = {}
    sort_order_dict = {}
    for idx, results_dir in enumerate(results_dirs_ordered):
        method_name = get_method_name(results_dir)
        while method_name in results_dirs_dict:
            method_name = add_number_to_method_name(method_name)
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
            if (results_dir / 'metadata.hjson').exists():
                results_dir_list = [results_dir]
            else:
                results_dir_list = []
                for d in results_dir.iterdir():
                    if (d / 'metadata.hjson').exists():
                        results_dir_list.append(d)

            for results_dir_i in results_dir_list:
                metadata = get_metadata(results_dir_i)
                scenario = get_scenario(metadata['planner_params']['scenario'])

                # NOTE: even though this is slow, parallelizing is not easy because "scenario" cannot be pickled
                metrics_filenames = list(results_dir_i.glob("*_metrics.pkl.gz"))
                for file_idx, metrics_filename in enumerate(metrics_filenames):
                    datum = load_gzipped_pickle(metrics_filename)
                    index_tuples.append([method_name, file_idx])
                    data.append([metric_func(scenario, metadata, datum) for metric_func in
                                 link_bot_planning.analysis.results_metrics.metrics_funcs])

        index = pd.MultiIndex.from_tuples(index_tuples, names=["method_name", "file_idx"])
        metrics = pd.DataFrame(data=data, index=index, columns=metrics_names)

        with pickle_filename.open("wb") as pickle_file:
            pickle.dump(metrics, pickle_file)
        rospy.loginfo(Fore.GREEN + f"Pickling metrics to {pickle_filename}" + Fore.RESET)

    return method_names, metrics


def get_metrics(args, out_dir, planning_results_dirs, get_method_name: Callable, get_metadata: Callable):
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
                    data.append([metric_func(scenario, metadata, datum) for metric_func in
                                 link_bot_planning.analysis.results_metrics.metrics_funcs])

        index = pd.MultiIndex.from_tuples(index_tuples, names=["method_name", "iteration_idx", "file_idx"])
        metrics = pd.DataFrame(data=data, index=index, columns=metrics_names)

        with pickle_filename.open("wb") as pickle_file:
            pickle.dump(metrics, pickle_file)
        rospy.loginfo(Fore.GREEN + f"Pickling metrics to {pickle_filename}" + Fore.RESET)

    return method_names, metrics


def load_fig_specs(analysis_params, figures_config: pathlib.Path):
    figures_config = load_hjson(figures_config)
    figspecs = []
    for fig_config in figures_config:
        figure_type = eval(fig_config.pop('type'))
        reductions = fig_config.pop('reductions')
        axes_names = DEFAULT_AXES_NAMES

        fig = figure_type(analysis_params, **fig_config)

        figspec = FigSpec(fig=fig, reductions=reductions, axes_names=axes_names)
        figspecs.append(figspec)
    return figspecs


def load_table_specs(tables_config: pathlib.Path, table_format: str):
    tables_conf = load_hjson(tables_config)
    tablespecs = []
    for table_conf in tables_conf:
        table_type = eval(table_conf.pop('type'))
        reductions = table_conf.pop('reductions')
        axes_names = DEFAULT_AXES_NAMES

        table = table_type(table_format=table_format, **table_conf)

        tablespec = TableSpec(table=table, reductions=reductions, axes_names=axes_names)
        tablespecs.append(tablespec)
    return tablespecs


def reduce_metrics3(reductions: List[List], metrics: pd.DataFrame):
    reduced_metrics = []
    for reduction in reductions:
        metric_i = metrics.copy()
        for reduction_step in reduction:
            group_by, metric, agg = reduction_step
            assert metric is not None
            if group_by is None or len(group_by) == 0:
                metric_i = metric_i.agg({metric: agg})
            elif group_by is not None and agg is not None:
                metric_i = metric_i.groupby(group_by).agg({metric: agg})
            elif group_by is not None and agg is None:
                metric_i.set_index(group_by, inplace=True)
                metric_i = metric_i[metric]
            else:
                raise NotImplementedError()
        reduced_metrics.append(metric_i)

    reduced_metrics = pd.concat(reduced_metrics, axis=1)
    return reduced_metrics


def load_results2(results_dirs: List[pathlib.Path], regenerate: bool):
    dfs = []
    for d in progressbar(results_dirs, widgets=mywidgets):
        data_filenames = list(d.glob("*_metrics.pkl.gz"))
        df_filename = d / 'df.pkl'
        metadata_filename = d / 'metadata.hjson'
        metadata = load_hjson(metadata_filename)
        if not df_filename.exists() or regenerate:
            scenario = get_scenario_cached(metadata['planner_params']['scenario'])
            data = []
            print()
            halo = Halo()
            halo.start()
            for data_filename in data_filenames:
                datum = load_gzipped_pickle(data_filename)
                row = make_row(datum, metadata, scenario)
                data.append(row)
            df_i = pd.DataFrame(data)
            halo.stop()
            with df_filename.open("wb") as f:
                pickle.dump(df_i, f)
        else:
            with df_filename.open("rb") as f:
                df_i = pickle.load(f)
        dfs.append(df_i)

    df = pd.concat(dfs)
    df.columns = column_names
    return df


def make_row(datum, metadata, scenario):
    metrics_values = [metric_func(scenario, metadata, datum) for metric_func in metrics_funcs]
    trial_idx = datum['trial_idx']
    try:
        seed_guess = datum['steps'][0]['planning_query'].seed - 100000 * trial_idx
    except KeyError:
        seed_guess = 0
    seed = datum.get('seed', seed_guess)
    row = [
        metadata['planner_params']['method_name'],
        seed,
        metadata.get('ift_iteration', 0),
        trial_idx,
        datum['uuid'],
    ]
    row += metrics_values
    return row


def load_results(df, results_dirs: List[pathlib.Path], outfile):
    for metadata, datum in progressbar(PlanningResultsGenerator(results_dirs), widgets=mywidgets):
        already_exists = datum['uuid'] in df['uuid'].unique()
        if already_exists:
            continue
        # Assume we don't need separate instances of the scenario every time
        scenario = get_scenario_cached(metadata['planner_params']['scenario'])
        row = make_row(datum, metadata, scenario)
        df = df_append(df, row)

    # if everything went well now overwrite the input file
    with outfile.open("wb") as f:
        pickle.dump(df, f)
    print(Fore.MAGENTA + f"Wrote {outfile.as_posix()}" + Fore.RESET)
    return df


class PlanningResultsGenerator:

    def __init__(self, results_dirs: List[pathlib.Path]):
        self.metadata_and_filenames = []
        for d in results_dirs:
            data_filenames = list(
                d.glob("*_metrics.pkl.gz"))  # FIXME: "metrics" here is a misleading naming convention :(
            metadata_filename = d / 'metadata.hjson'
            metadata = load_hjson(metadata_filename)
            # for data_filename in data_filenames[:10]:
            for data_filename in data_filenames:
                self.metadata_and_filenames.append((metadata, data_filename))

    def __len__(self):
        return len(self.metadata_and_filenames)

    def __iter__(self):
        for metadata, data_filename in self.metadata_and_filenames:
            datum = load_gzipped_pickle(data_filename)
            yield metadata, datum
