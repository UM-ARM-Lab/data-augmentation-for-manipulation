#!/usr/bin/env python
import argparse
import pathlib

import matplotlib.pyplot as plt
import seaborn as sns

from analysis.analyze_results import planning_results
from analysis.results_figures import violinplot
from arc_utilities import ros_init
from link_bot_pycommon.string_utils import shorten
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(None)


def analyze_planning_results(args):
    outdir, df, table_format = planning_results(args.results_dirs, args.regenerate)
    df = df.sort_values("method_name", ascending=False)

    print(df[['any_solved']].to_string(index=False))
    print(df[['success']].to_string(index=False))

    successes = (df['success'] == 1).sum()
    total = df['success'].count()
    print(f"{successes}/{total} = {successes / total}")

    _, ax = violinplot(df, outdir, 'method_name', 'task_error', "Task Error", save=False)
    _, ymax = ax.get_ylim()
    ax.set_ylim([0, ymax])
    ax.set_ylabel('task error')
    ax.set_xlabel('')
    plt.savefig(outdir / f'real_robot_task_error.png')

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.barplot(
        ax=ax,
        data=df,
        x='method_name',
        y='success',
        palette='colorblind',
        linewidth=5,
        ci=None,
    )
    for p in ax.patches:
        _x = p.get_x() + p.get_width() / 2
        _y = p.get_y() + p.get_height() + 0.02
        value = '{:.2f}'.format(p.get_height())
        ax.text(_x, _y, value, ha="center")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel('')
    ax.set_title('Success Rate')
    plt.savefig(outdir / f'real_robot_success.png')

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
