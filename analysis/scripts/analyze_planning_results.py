#!/usr/bin/env python
import argparse
import pathlib

import matplotlib.pyplot as plt
import seaborn as sns

from analysis.analyze_results import planning_results
from analysis.results_figures import violinplot
from arc_utilities import ros_init
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(0.1)


def metrics_main(args):
    outdir, df, table_format = planning_results(args.results_dirs, args.regenerate)

    violinplot(df, outdir, 'method_name', 'task_error', "Task Error")
    violinplot(df, outdir, 'method_name', 'normalized_model_error', 'Normalized Model Error')

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

    if not args.no_plot:
        plt.show()


@ros_init.with_ros("analyse_planning_results")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dirs', help='results directory', type=pathlib.Path, nargs='+')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--regenerate', action='store_true')
    parser.add_argument('--style', default='slides')
    parser.set_defaults(func=metrics_main)

    args = parser.parse_args()

    plt.style.use(args.style)

    metrics_main(args)


if __name__ == '__main__':
    import numpy as np

    np.seterr(all='raise')  # DEBUGGING
    main()
