#!/usr/bin/env python
import argparse
import pathlib

import matplotlib.pyplot as plt
import tabulate

from analysis.analyze_results import load_table_specs, load_planning_results, generate_tables, planning_results
from analysis.results_utils import get_all_results_subdirs
from arc_utilities import ros_init
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(0.1)


def metrics_main(args):
    outdir, df, table_format = planning_results(args.results_dirs, args.regenerate, args.latex)

    table_specs = load_table_specs(args.tables_config, table_format)

    generate_tables(df, outdir, table_specs)


@ros_init.with_ros("analyse_planning_results")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dirs', help='results directory', type=pathlib.Path, nargs='+')
    parser.add_argument('--figures-config', type=pathlib.Path,
                        default=pathlib.Path("figures_configs/planning_evaluation.hjson"))
    parser.add_argument('--tables-config', type=pathlib.Path,
                        default=pathlib.Path("tables_configs/factors.hjson"))
    parser.add_argument('--analysis-params', type=pathlib.Path,
                        default=pathlib.Path("analysis_params/env_across_methods.json"))
    parser.add_argument('--no-plot', action='store_true')
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
    import numpy as np

    np.seterr(all='raise')  # DEBUGGING
    main()
