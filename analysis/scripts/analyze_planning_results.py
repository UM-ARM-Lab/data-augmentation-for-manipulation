#!/usr/bin/env python
import argparse
import pathlib

import matplotlib.pyplot as plt
import seaborn as sns

from analysis.analyze_results import planning_results
from analysis.results_figures import barplot, boxplot
from arc_utilities import ros_init
from link_bot_pycommon.string_utils import shorten
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(None)


def analyze_planning_results(args):
    outdir, df, table_format = planning_results(args.results_dirs, args.regenerate)

    def _shorten(c):
        return shorten(c.split('/')[0])[:16]

    df['x_name'] = df['classifier_name'].map(_shorten)

    # print(df[['any_solved']].to_string(index=False))
    # print(df[['success']].to_string(index=False))

    successes = (df['success'] == 1).sum()
    total = df['success'].count()
    print(f"{successes}/{total} = {successes / total}")

    hue = 'method_name'

    _, ax = boxplot(df, outdir, hue, 'task_error', "Task Error", figsize=(12, 8))
    ax.axhline(y=0.045, linestyle='--')
    ax.set_ylim([0, 0.4])
    plt.savefig(outdir / f'task_error.png')

    boxplot(df, outdir, hue, 'normalized_model_error', "Model Error", figsize=(12, 8))

    barplot(df, outdir, hue, 'any_solved', "Any Plans Found?", figsize=(12, 8), ci=None)

    success_barplot(df, 'success', hue, outdir, figsize=(12, 8))

    _, ax = boxplot(df, outdir, hue, 'task_error_given_solved', "Task Error (given solved)", figsize=(12, 8))
    ax.axhline(y=0.045, linestyle='--')
    ax.set_ylim([0, 0.4])
    plt.savefig(outdir / f'task_error_given_solved.png')

    success_barplot(df, 'success_given_solved', hue, outdir, figsize=(12, 8))

    boxplot(df, outdir, hue, 'planning_time', "Total Planning Time", figsize=(12, 8))

    boxplot(df, outdir, hue, 'num_actions', "# Actions", figsize=(12, 8))

    if not args.no_plot:
        plt.show(block=True)


def success_barplot(df, y, hue, outdir, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(ax=ax, data=df, x=hue, y=y, palette='colorblind', linewidth=5, ci=None)
    for p in ax.patches:
        _x = p.get_x() + p.get_width() / 2
        _y = p.get_y() + p.get_height() + 0.02
        value = '{:.2f}'.format(p.get_height())
        ax.text(_x, _y, value, ha="center")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{y}")
    plt.savefig(outdir / f'{y}.png')


@ros_init.with_ros("analyse_planning_results")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dirs', help='results directory', type=pathlib.Path, nargs='+')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--regenerate', action='store_true')
    parser.add_argument('--style', default='slides')

    args = parser.parse_args()

    # plt.style.use(args.style)

    analyze_planning_results(args)


if __name__ == '__main__':
    main()
