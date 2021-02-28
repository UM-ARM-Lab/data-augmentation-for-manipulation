#!/usr/bin/env python
import argparse
import pickle

import colorama
import hjson
from colorama import Style
from tabulate import tabulate

import rospy
from arc_utilities.filesystem_utils import get_all_subfolders
from link_bot_planning.results_metrics import *
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.metric_utils import dict_to_pvalue_table
from link_bot_pycommon.pycommon import paths_from_json
from link_bot_pycommon.serialization import my_hdump, load_gzipped_pickle
from moonshine.filepath_tools import load_json_or_hjson
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(0.1)


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

    tables_filename = out_dir / 'tables.txt'
    with tables_filename.open("w") as tables_file:
        tables_file.truncate()

    sort_order_dict = {}
    for sort_idx, subfolder in enumerate(subfolders_ordered):
        metadata = load_json_or_hjson(subfolder, 'metadata')
        method_name = metadata['planner_params'].get('method_name', subfolder.name)
        sort_order_dict[method_name] = sort_idx

    pickle_filename = out_dir / f"{unique_comparison_name}-metrics.pkl"
    if pickle_filename.exists() and not args.regenerate:
        rospy.loginfo(Fore.GREEN + f"Loading existing metrics from {pickle_filename}")
        with pickle_filename.open("rb") as pickle_file:
            metrics: Dict[type, ResultsMetric] = pickle.load(pickle_file)

        # update the analysis params so we don't need to regenerate metrics
        for metric in metrics:
            metric.params = analysis_params

        with pickle_filename.open("wb") as pickle_file:
            pickle.dump(metrics, pickle_file)
        rospy.loginfo(Fore.GREEN + f"Pickling metrics to {pickle_filename}")
    else:
        rospy.loginfo(Fore.GREEN + f"Generating metrics")
        metrics = generate_metrics(analysis_params, out_dir, subfolders_ordered)

        with pickle_filename.open("wb") as pickle_file:
            pickle.dump(metrics, pickle_file)
        rospy.loginfo(Fore.GREEN + f"Pickling metrics to {pickle_filename}")

    figures = [
        TaskErrorLineFigure(analysis_params, metrics[TaskError]),
        violin_plot(analysis_params, metrics[TaskError], 'Task Error'),
        box_plot(analysis_params, metrics[NRecoveryActions], "Num Recovery Actions"),
        box_plot(analysis_params, metrics[TotalTime], 'Total Time'),
        violin_plot(analysis_params, metrics[TotalTime], 'Total Time'),
        box_plot(analysis_params, metrics[NPlanningAttempts], 'Num Planning Attempts'),
        box_plot(analysis_params, metrics[NMERViolations], 'Num MER Violations'),
        box_plot(analysis_params, metrics[PlanningTime], 'Planning Time'),
        box_plot(analysis_params, metrics[PercentageMERViolations], '% MER Violations'),
        BarChartPercentagePerMethodFigure(analysis_params, metrics[PlannerSolved], '% Planner Returned Solved'),
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

        # For saving metrics since this script is kind of slow it's nice to save the output
        with tables_filename.open("a") as tables_file:
            tables_file.write(figure.name)
            tables_file.write('\n')
            tables_file.write(table)
            tables_file.write('\n')

    for figure in figures:
        pvalue_table_title = f"p-value matrix [{figure.name}]"
        pvalue_table = dict_to_pvalue_table(figure.metric.values, table_format=table_format)
        print(Style.BRIGHT + pvalue_table_title + Style.NORMAL)
        print(pvalue_table)
        with tables_filename.open("a") as tables_file:
            tables_file.write(pvalue_table_title)
            tables_file.write('\n')
            tables_file.write(pvalue_table)
            tables_file.write('\n')

    for figure in figures:
        figure.make_figure()
        figure.save_figure()

    if not args.no_plot:
        plt.show()


def generate_metrics(analysis_params: Dict, out_dir: pathlib.Path, subfolders_ordered: List):
    metrics = {}

    def _include_metric(metric: ResultsMetric):
        metrics[metric.__class__] = metric

    _include_metric(TaskError(analysis_params, results_dir=out_dir))
    _include_metric(TaskError(analysis_params, results_dir=out_dir))
    _include_metric(NRecoveryActions(analysis_params, results_dir=out_dir))
    _include_metric(TotalTime(analysis_params, results_dir=out_dir))
    _include_metric(NPlanningAttempts(analysis_params, results_dir=out_dir))
    _include_metric(NMERViolations(analysis_params, results_dir=out_dir))
    _include_metric(PlanningTime(analysis_params, results_dir=out_dir))
    _include_metric(PercentageMERViolations(analysis_params, results_dir=out_dir))
    _include_metric(PlannerSolved(analysis_params, results_dir=out_dir))

    for subfolder in subfolders_ordered:
        metrics_filenames = list(subfolder.glob("*_metrics.pkl.gz"))

        metadata = load_json_or_hjson(subfolder, 'metadata')

        method_name = metadata['planner_params'].get('method_name', subfolder.name)
        scenario = get_scenario(metadata['scenario'])

        for metric in metrics.values():
            metric.setup_method(method_name, metadata)

        # NOTE: even though this is slow, parallelizing is not easy because "scenario" cannot be pickled
        for plan_idx, metrics_filename in enumerate(metrics_filenames):
            datum = load_gzipped_pickle(metrics_filename)
            for metric in metrics.values():
                metric.aggregate_trial(method_name, scenario, datum)

        for metric in metrics.values():
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
    parser.add_argument('--show-all-trials', action='store_true')
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--regenerate', action='store_true')
    parser.add_argument('--debug', action='store_true', help='will only run on a few examples to speed up debugging')
    parser.add_argument('--style', default='slides')
    parser.set_defaults(func=metrics_main)

    args = parser.parse_args()

    plt.style.use(args.style)

    metrics_main(args)


if __name__ == '__main__':
    main()
