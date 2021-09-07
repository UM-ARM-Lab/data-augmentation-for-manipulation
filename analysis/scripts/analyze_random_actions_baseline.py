#!/usr/bin/env python
import argparse
import pathlib
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis.analyze_results import planning_results
from analysis.results_figures import lineplot
from arc_utilities import ros_init
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(0.1)


def metrics_main(args):
    outdir, df, _ = planning_results(args.results_dirs, args.regenerate, args.latex)

    # turn classifier_dataset into an iter number
    def _dataset_dir_to_iter(p):
        p = pathlib.Path(p)
        for part in p.parts:
            m = re.match(r'iter_(\d+)', part)
            if m:
                i = int(m.group(1))
                return i
        return -1

    iter_key = 'classifier_dataset_iter'
    df[iter_key] = df['classifier_dataset'].map(_dataset_dir_to_iter)

    df = df.groupby([iter_key]).agg('mean').reset_index()

    w = 10
    x = lineplot(df, iter_key, 'success', 'Success Rate', iter_key)
    x.set_ylim(0, 1)
    plt.savefig(outdir / f'success_rate.png')
    lineplot(df, iter_key, 'task_error', 'Task Error', iter_key)
    lineplot(df, iter_key, 'normalized_model_error', 'Normalized Model Error', iter_key)

    if not args.no_plot:
        plt.show()

    # generate_tables(df, outdir, table_specs)


@ros_init.with_ros("analyse_random_actions_baseline")
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
