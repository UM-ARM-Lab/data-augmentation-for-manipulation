#!/usr/bin/env python
import argparse
import pathlib
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis.analyze_results import planning_results
from analysis.results_figures import lineplot
from arc_utilities import ros_init
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(0.1)


def metrics_main(args):
    outdir, df, table_specs = planning_results(args.results_dirs, args.regenerate, args.latex)

    iter_key = 'ift_iteration'
    df = df.groupby([iter_key, 'used_augmentation']).agg('mean').reset_index()

    w = 10
    x = lineplot(df, iter_key, 'success', 'Success Rate', hue='used_augmentation')
    x.set_ylim(-0.01, 1.01)
    plt.savefig(outdir / f'success_rate.png')
    x = lineplot(df, iter_key, 'success', 'Success Rate (rolling)', window=w, hue='used_augmentation')
    x.set_ylim(-0.01, 1.01)
    plt.savefig(outdir / f'success_rate_rolling.png')
    lineplot(df, iter_key, 'task_error', 'Task Error', hue='used_augmentation')
    lineplot(df, iter_key, 'task_error', 'Task Error (rolling)', window=w, hue='used_augmentation')
    lineplot(df, iter_key, 'normalized_model_error', 'Normalized Model Error', hue='used_augmentation')
    plt.savefig(outdir / f'normalized_model_error.png')
    lineplot(df, iter_key, 'normalized_model_error', 'Normalized Model Error (rolling)', window=w, hue='used_augmentation')
    plt.savefig(outdir / f'normalized_model_error_rolling.png')

    if not args.no_plot:
        plt.show()

    # generate_tables(df, outdir, table_specs)

@ros_init.with_ros("analyse_ift_results")
def main():
    pd.options.display.max_rows = 999

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
    plt.rcParams['figure.figsize'] = (20, 10)
    sns.set(rc={'figure.figsize': (7, 4)})

    metrics_main(args)


if __name__ == '__main__':
    import numpy as np

    np.seterr(all='raise')  # DEBUGGING
    main()
