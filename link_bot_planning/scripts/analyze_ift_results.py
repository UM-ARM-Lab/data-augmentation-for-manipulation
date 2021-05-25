#!/usr/bin/env python
import argparse
import pathlib

import colorama
import matplotlib.pyplot as plt
import numpy as np

import rospy
from link_bot_planning.analysis.analyze_results import get_metrics, load_fig_specs, load_table_specs
from link_bot_planning.analysis.figspec import get_data_for_figure, get_data_for_table
# noinspection PyUnresolvedReferences
from link_bot_planning.analysis.results_figures import *
from link_bot_planning.analysis.results_metrics import *
from link_bot_pycommon.args import my_formatter
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

    def _get_metadata(results_dir: pathlib.Path):
        return load_json_or_hjson(results_dir.parent.parent, 'logfile')

    def _get_method_name(results_dir: pathlib.Path):
        log = load_hjson(results_dir.parent / 'logfile.hjson')
        from_env = log['from_env']
        to_env = log['to_env']
        if args.no_use_nickname:
            return f'{from_env}_to_{to_env}'
        return log['nickname']

    method_names, metrics = get_metrics(args, out_dir, planning_results_dirs, _get_method_name, _get_metadata)

    # Figures & Tables
    fig_specs = load_fig_specs(analysis_params, args)
    table_specs = load_table_specs(analysis_params, args, table_format)

    for spec in fig_specs:
        data_for_figure = get_data_for_figure(spec, metrics)

        spec.fig.make_figure(data_for_figure, method_names)
        spec.fig.save_figure(out_dir)

    for spec in table_specs:
        data_for_table = get_data_for_table(spec, metrics)

        spec.table.make_table(data_for_table, method_names)
        spec.table.save(out_dir)

    if not args.no_plot:
        for spec in table_specs:
            spec.table.print()

        for spec in fig_specs:
            spec.fig.fig.set_tight_layout(True)
        plt.show()


def main():
    colorama.init(autoreset=True)

    rospy.init_node("analyse_ift_results")
    np.set_printoptions(suppress=True, precision=4, linewidth=220)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('ift_dirs', help='results directory', type=pathlib.Path, nargs='+')
    parser.add_argument('--figures-config', type=pathlib.Path,
                        default=pathlib.Path("analysis_params/figures_configs/ift.hjson"))
    parser.add_argument('--tables-config', type=pathlib.Path,
                        default=pathlib.Path("analysis_params/tables_configs/ift.hjson"))
    parser.add_argument('--analysis-params', type=pathlib.Path,
                        default=pathlib.Path("analysis_params/env_across_methods.json"))
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--no-use-nickname', action='store_true')
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
