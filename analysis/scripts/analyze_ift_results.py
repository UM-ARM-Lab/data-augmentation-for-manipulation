#!/usr/bin/env python
import argparse
import pathlib

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

    w = 10
    iter_key = 'ift_iteration'

    z = df.groupby([iter_key, 'used_augmentation', 'seed', 'ift_uuid']).agg('mean').reset_index(iter_key)
    x = lineplot(z, iter_key, 'success', 'Success Rate [all separate] (rolling)', window=w, hue='ift_uuid')
    x.set_xlim(-0.01, 100.01)
    x.set_ylim(-0.01, 1.01)

    z2 = df.groupby([iter_key]).agg('mean').reset_index(iter_key)
    x = lineplot(z2, iter_key, 'success', 'Success Rate [all combined] (rolling)', window=w)
    x.set_xlim(-0.01, 100.01)
    x.set_ylim(-0.01, 1.01)

    z3 = df.groupby([iter_key, 'used_augmentation', 'seed']).agg('mean').reset_index(iter_key)

    x = lineplot(z3, iter_key, 'success', 'Success Rate', hue='used_augmentation')
    x.set_xlim(-0.01, 100.01)
    x.set_ylim(-0.01, 1.01)
    plt.savefig(outdir / f'success_rate.png')

    x = lineplot(z3, iter_key, 'success', 'Success Rate (rolling)', window=w, hue='used_augmentation')
    x.set_xlim(-0.01, 100.01)
    x.set_ylim(-0.01, 1.01)
    plt.savefig(outdir / f'success_rate_rolling.png')

    x = lineplot(z3, iter_key, 'task_error', 'Task Error', hue='used_augmentation')
    x.set_xlim(-0.01, 100.01)

    x = lineplot(z3, iter_key, 'task_error', 'Task Error (rolling)', window=w, hue='used_augmentation')
    x.set_xlim(-0.01, 100.01)

    x = lineplot(z3, iter_key, 'normalized_model_error', 'Normalized Model Error', hue='used_augmentation')
    x.set_xlim(-0.01, 100.01)
    plt.savefig(outdir / f'normalized_model_error.png')

    x = lineplot(z3, iter_key, 'normalized_model_error', 'Normalized Model Error (rolling)', window=w,
                 hue='used_augmentation')
    x.set_xlim(-0.01, 100.01)
    plt.savefig(outdir / f'normalized_model_error_rolling.png')

    if not args.no_plot:
        plt.show()

    # generate_tables(z3, outdir, table_specs)


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
