#!/usr/bin/env python
import argparse
import pathlib

import matplotlib.pyplot as plt
import seaborn as sns

from analysis.analyze_results import planning_results
from arc_utilities import ros_init
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(0.1)


def metrics_main(args):
    outdir, df, table_format = planning_results(args.results_dirs, args.regenerate, args.latex)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.violinplot(
        ax=ax,
        data=df,
        x='method_name',
        y='task_error',
        linewidth=5,
    )
    ax.set_title('Task Error')
    plt.savefig(outdir / 'task_error.png')

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(
        ax=ax,
        data=df,
        x='method_name',
        y='success',
        linewidth=5,
        ci=None,
    )
    ax.set_title('Success')
    ax.set_ylim(-0.01, 1.01)
    plt.savefig(outdir / 'success.png')

    plt.show()


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
