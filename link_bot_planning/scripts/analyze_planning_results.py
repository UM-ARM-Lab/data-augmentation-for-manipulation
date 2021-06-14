#!/usr/bin/env python
import argparse
import pathlib

import matplotlib.pyplot as plt

from arc_utilities import ros_init
from arc_utilities.filesystem_utils import get_all_subdirs
from link_bot_planning.analysis.analyze_results import load_fig_specs, get_metrics2, load_table_specs
from link_bot_planning.analysis.figspec import get_data_for_figure, get_data_for_table
from link_bot_planning.analysis.results_metrics import load_analysis_params
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
        table_format = 'simple'

    def _get_metadata(results_dir: pathlib.Path):
        return load_json_or_hjson(results_dir, 'metadata')

    def _get_method_name(results_dir: pathlib.Path):
        method_name_filename = results_dir / 'method_name'
        if method_name_filename.exists():
            with method_name_filename.open("r") as mnf:
                return mnf.readline().strip("\n")
        metadata_filename = results_dir / 'metadata.hjson'
        if not metadata_filename.exists():
            metadata_filename = list(results_dir.iterdir())[0] / 'metadata.hjson'
        metadata = load_hjson(metadata_filename)
        return metadata['planner_params']['method_name']

    results_dirs = get_all_subdirs(args.results_dirs)
    method_names, metrics = get_metrics2(args, out_dir, results_dirs, _get_method_name, _get_metadata)

    # Figures & Tables
    figspecs = load_fig_specs(analysis_params, args)
    table_specs = load_table_specs(analysis_params, args, table_format)

    for spec in figspecs:
        data_for_figure = get_data_for_figure(spec, metrics)

        spec.fig.make_figure(data_for_figure, method_names)
        spec.fig.save_figure(out_dir)

    for spec in table_specs:
        data_for_table = get_data_for_table(spec, metrics)

        spec.table.make_table(data_for_table, method_names)
        spec.table.save(out_dir)

    for spec in table_specs:
        print()
        spec.table.print()
        print()

    if not args.no_plot:
        for spec in figspecs:
            spec.fig.fig.set_tight_layout(True)
        plt.show()


@ros_init.with_ros("analyse_planning_results")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dirs', help='results directory', type=pathlib.Path, nargs='+')
    parser.add_argument('--figures-config', type=pathlib.Path,
                        default=pathlib.Path("analysis_params/figures_configs/planning_evaluation.hjson"))
    parser.add_argument('--tables-config', type=pathlib.Path,
                        default=pathlib.Path("analysis_params/tables_configs/planning_evaluation.hjson"))
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
