#!/usr/bin/env python
import argparse
import pathlib

import matplotlib.pyplot as plt
import seaborn as sns
import tabulate

from analysis.analyze_results import load_table_specs, load_planning_results, generate_tables
from analysis.results_metrics import load_analysis_params
from analysis.results_utils import get_all_results_subdirs
from arc_utilities import ros_init
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
        table_format = tabulate.simple_separated_format("\t")

    results_dirs = get_all_results_subdirs(args.results_dirs)
    df = load_planning_results(results_dirs, regenerate=args.regenerate)
    df.to_csv("/media/shared/analysis/tmp_results.csv")

    table_specs = load_table_specs(args.tables_config, table_format)

    lineplot(df, 'ift_iteration', 'success', 'Success Rate', out_dir)
    lineplot(df, 'ift_iteration', 'success', 'Success Rate (rolling)', out_dir, window=5)
    lineplot(df, 'ift_iteration', 'task_error', 'Task Error', out_dir)
    lineplot(df, 'ift_iteration', 'task_error', 'Task Error (rolling)', out_dir, window=5)

    plt.show()

    generate_tables(df, out_dir, table_specs)


def lineplot(df, x: str, metric: str, title: str, outdir: pathlib.Path, window=1):
    df = df.copy()
    df[metric] = df[metric].rolling(window=window, min_periods=1).agg('mean')
    plt.figure()
    ax = sns.lineplot(
        data=df,
        x=x,
        y=metric,
        palette='colorblind',
        estimator='mean',
        ci='sd',
    ).set_title(title)
    outfilename = outdir / f'{title}.png'
    plt.savefig(outfilename)
    return ax


@ros_init.with_ros("analyse_ift_results")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dirs', help='results directory', type=pathlib.Path, nargs='+')
    parser.add_argument('--tables-config', type=pathlib.Path,
                        default=pathlib.Path("tables_configs/planning_evaluation.hjson"))
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
