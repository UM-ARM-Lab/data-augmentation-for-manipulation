#!/usr/bin/env python
import argparse
import pathlib

import colorama
import matplotlib.pyplot as plt
import numpy as np

import rospy
from arc_utilities import ros_init
from link_bot_planning.analysis.analyze_results import get_metrics, load_figspecs
from link_bot_planning.analysis.figspec import get_data_for_figure
from link_bot_planning.analysis.results_metrics import load_analysis_params
from link_bot_pycommon.args import my_formatter
from moonshine.filepath_tools import load_hjson, load_json_or_hjson
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(0.1)


def metrics_main(args):
    analysis_params = load_analysis_params(args.analysis_params)

    # The default for where we write results
    out_dir = args.results_dirs[0]
    print(f"Writing analysis to {out_dir}")

    if args.latex:
        table_format = 'latex_raw'
    else:
        table_format = 'fancy_grid'

    def _get_metadata(results_dir: pathlib.Path):
        return load_json_or_hjson(results_dir, 'metadata')

    def _get_method_name(results_dir: pathlib.Path):
        metadata = load_hjson(results_dir / 'metadata.hjson')
        return metadata['planner_params']['method_name']

    method_names, metrics = get_metrics(args, out_dir, args.results_dirs, _get_method_name, _get_metadata)

    # Figures & Tables
    figspecs = load_figspecs(analysis_params, args)

    for spec in figspecs:
        data_for_figure = get_data_for_figure(spec, metrics)

        spec.fig.make_figure(data_for_figure, method_names)
        spec.fig.save_figure(out_dir)

    # make_tables(tables, analysis_params, sort_order_dict, table_format, tables_filename)

    if not args.no_plot:
        for spec in figspecs:
            spec.fig.fig.set_tight_layout(True)
        plt.show()


@ros_init.with_ros("analyse_planning_results")
def main():
    colorama.init(autoreset=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('results_dirs', help='results directory', type=pathlib.Path, nargs='+')
    parser.add_argument('--figures-config', type=pathlib.Path,
                        default=pathlib.Path("analysis_params/figures_configs/ift.hjson"))
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
