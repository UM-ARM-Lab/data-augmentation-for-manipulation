#!/usr/bin/env python
import argparse
import pathlib

import matplotlib.pyplot as plt

from analysis.analyze_results import planning_results
from analysis.results_figures import violinplot, barplot
from arc_utilities import ros_init
from link_bot_pycommon.string_utils import shorten
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(0.1)


def analyze_planning_results(args):
    outdir, df, table_format = planning_results(args.results_dirs, args.regenerate)

    def _shorten(c):
        return shorten(c.split('/')[0])[:16]

    df['x_name'] = df['classifier_name'].map(_shorten) + '-' + df['accept_type']

    print(df['any_solved'])

    _, ax = violinplot(df, outdir, 'x_name', 'task_error', "Task Error", hue='accept_type')
    _, ymax = ax.get_ylim()
    ax.set_ylim([0, ymax])
    _, ax = violinplot(df, outdir, 'x_name', 'normalized_model_error', 'Normalized Model Error', hue='accept_type')
    _, ymax = ax.get_ylim()
    ax.set_ylim([0, ymax])
    _, ax = violinplot(df, outdir, 'x_name', 'combined_error', 'Combined Error', hue='accept_type')
    _, ymax = ax.get_ylim()
    ax.set_ylim([0, ymax])

    z = df.groupby("method_name").agg({
        'success': 'mean',
        'accept_type': 'first',
        'x_name': 'first',
    })
    _, ax = barplot(z, outdir, x='x_name', y='success', title='Success', hue='accept_type')
    ax.set_ylim(-0.01, 1.01)

    if not args.no_plot:
        plt.show()


@ros_init.with_ros("analyse_planning_results")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dirs', help='results directory', type=pathlib.Path, nargs='+')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--regenerate', action='store_true')
    parser.add_argument('--style', default='slides')

    args = parser.parse_args()

    plt.style.use(args.style)

    analyze_planning_results(args)


if __name__ == '__main__':
    main()
