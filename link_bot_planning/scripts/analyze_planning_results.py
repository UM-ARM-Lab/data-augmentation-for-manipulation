#!/usr/bin/env python
import argparse
import gzip
import json
import pickle

import colorama
import hjson
import orjson
from colorama import Style
from tabulate import tabulate

import rospy
from arc_utilities.filesystem_utils import get_all_subfolders
from link_bot_planning.results_metrics import *
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.metric_utils import dict_to_pvalue_table
from link_bot_pycommon.pycommon import paths_from_json
from link_bot_pycommon.serialization import my_hdump


def save_order(outdir: pathlib.Path, subfolders_ordered: List[pathlib.Path]):
    sort_order_filename = outdir / 'sort_order.txt'
    with sort_order_filename.open("w") as sort_order_file:
        my_hdump(subfolders_ordered, sort_order_file)


def load_sort_order(outdir: pathlib.Path, unsorted_dirs: List[pathlib.Path]):
    sort_order_filename = outdir / 'sort_order.txt'
    if sort_order_filename.exists():
        with sort_order_filename.open("r") as sort_order_file:
            subfolders_ordered = hjson.load(sort_order_file)
        subfolders_ordered = paths_from_json(subfolders_ordered)
        return subfolders_ordered
    return unsorted_dirs


def metrics_main(args):
    with args.analysis_params.open('r') as analysis_params_file:
        analysis_params = hjson.load(analysis_params_file)

    # The default for where we write results
    out_dir = args.results_dirs[0]
    print(f"Writing analysis to {out_dir}")

    unique_comparison_name = "-".join([p.name for p in args.results_dirs])

    # For saving metrics since this script is kind of slow
    table_outfile = open(out_dir / 'tables.txt', 'w')

    subfolders = get_all_subfolders(args)

    if args.final:
        table_format = 'latex_raw'
        for subfolder_idx, subfolder in enumerate(subfolders):
            print("{}) {}".format(subfolder_idx, subfolder))
        sort_order = input(Fore.CYAN + "Enter the desired table order:\n" + Fore.RESET)
        subfolders_ordered = [subfolders[int(i)] for i in sort_order.split(' ')]
        save_order(out_dir, subfolders_ordered)
    else:

        table_format = 'fancy_grid'
        subfolders_ordered = load_sort_order(out_dir, subfolders)

    sort_order_dict = {}
    for sort_idx, subfolder in enumerate(subfolders_ordered):
        with (subfolder / 'metadata.json').open('r') as metadata_file:
            metadata = json.load(metadata_file)
        method_name = metadata['planner_params'].get('method_name', subfolder.name)
        sort_order_dict[method_name] = sort_idx

    pickle_filename = out_dir / f"{unique_comparison_name}-metrics.pkl"
    if pickle_filename.exists() and not args.regenerate:
        rospy.loginfo(Fore.GREEN + f"Loading existing metrics from {pickle_filename}")
        with pickle_filename.open("rb") as pickle_file:
            metrics: List[ResultsMetric] = pickle.load(pickle_file)

        # update the analysis params so we don't need to regenerate metrics
        for metric in metrics:
            metric.params = analysis_params

        with pickle_filename.open("wb") as pickle_file:
            pickle.dump(metrics, pickle_file)
        rospy.loginfo(Fore.GREEN + f"Pickling metrics to {pickle_filename}")
    else:
        rospy.loginfo(Fore.GREEN + f"Generating metrics")
        metrics = generate_metrics(args, out_dir, subfolders_ordered)

        with pickle_filename.open("wb") as pickle_file:
            pickle.dump(metrics, pickle_file)
        rospy.loginfo(Fore.GREEN + f"Pickling metrics to {pickle_filename}")

    figures = [
        # FinalExecutionToGoalErrorFigure(analysis_params, metrics[0]),
        # NRecoveryActions(analysis_params, metrics[1]),
        # TotalTime(analysis_params, metrics[2]),
        # NPlanningAttempts(analysis_params, metrics[3]),
        TaskErrorBoxplotFigure(analysis_params, metrics[0]),
    ]

    for figure in figures:
        figure.params = analysis_params
        figure.sort_methods(sort_order_dict)

    for figure in figures:
        figure.enumerate_methods()

    for figure in figures:
        table_header, table_data = figure.make_table(table_format)
        if table_data is None:
            continue
        print(Style.BRIGHT + figure.name + Style.NORMAL)
        table = tabulate(table_data,
                         headers=table_header,
                         tablefmt=table_format,
                         floatfmt='6.4f',
                         numalign='center',
                         stralign='left')
        print(table)
        print()
        table_outfile.write(figure.name)
        table_outfile.write('\n')
        table_outfile.write(table)
        table_outfile.write('\n')

    for figure in figures:
        pvalue_table_title = f"p-value matrix [{figure.name}]"
        pvalue_table = dict_to_pvalue_table(figure.metric.values, table_format=table_format)
        print(Style.BRIGHT + pvalue_table_title + Style.NORMAL)
        print(pvalue_table)
        table_outfile.write(pvalue_table_title)
        table_outfile.write('\n')
        table_outfile.write(pvalue_table)
        table_outfile.write('\n')

    for figure in figures:
        figure.make_figure()
        figure.save_figure()

    if not args.no_plot:
        plt.show()


def generate_metrics(args, out_dir, subfolders_ordered):
    metrics = [
        TaskError(args, results_dir=out_dir),
        NRecoveryActions(args, results_dir=out_dir),
        TotalTime(args, results_dir=out_dir),
        NPlanningAttempts(args, results_dir=out_dir),
    ]
    for subfolder in subfolders_ordered:
        metrics_filenames = list(subfolder.glob("*_metrics.json.gz"))

        with (subfolder / 'metadata.json').open('r') as metadata_file:
            metadata = json.load(metadata_file)
        method_name = metadata['planner_params'].get('method_name', subfolder.name)
        scenario = get_scenario(metadata['scenario'])

        for metric in metrics:
            metric.setup_method(method_name, metadata)

        datums = []
        for plan_idx, metrics_filename in enumerate(metrics_filenames):
            if args.debug and plan_idx > 3:
                break
            with gzip.open(metrics_filename, 'rb') as metrics_file:
                data_str = metrics_file.read()
            # orjson is twice as fast, and yes it really matters here.
            datum = orjson.loads(data_str.decode("utf-8"))
            datums.append(datum)

        # NOTE: even though this is slow, parallelizing is not easy because "scenario" cannot be pickled
        for metric in metrics:
            for datum in datums:
                metric.aggregate_trial(method_name, scenario, datum)

        for metric in metrics:
            metric.convert_to_numpy_arrays()
    return metrics


def main():
    colorama.init(autoreset=True)

    rospy.init_node("analyse_planning_results")
    np.set_printoptions(suppress=True, precision=4, linewidth=180)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('results_dirs', help='results directory', type=pathlib.Path, nargs='+')
    parser.add_argument('analysis_params', type=pathlib.Path)
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--regenerate', action='store_true')
    parser.add_argument('--debug', action='store_true', help='will only run on a few examples to speed up debugging')
    parser.add_argument('--style', default='paper')
    parser.set_defaults(func=metrics_main)

    args = parser.parse_args()

    plt.style.use(args.style)

    metrics_main(args)


if __name__ == '__main__':
    main()
