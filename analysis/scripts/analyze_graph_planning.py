#!/usr/bin/env python
import argparse
import pathlib
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis.analyze_results import make_row
from analysis.results_figures import violinplot
from arc_utilities import ros_init
from link_bot_pycommon.get_scenario import get_scenario_cached
from link_bot_pycommon.string_utils import shorten
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(None)


def load_graph_planning_results(results_dirs, regenerate):
    data = []
    for results_dir in results_dirs:
        for results_filename in results_dir.glob("trial_*.pkl"):
            with results_filename.open("rb") as f:
                datum = pickle.load(f)
            scenario = get_scenario_cached(datum['scenario'])
            metadata = datum['metadata']
            row = make_row(datum, results_filename, metadata, scenario)
            data.append(datum)

    return pd.DataFrame(data, columns=[])


def analyze_graph_planning(results_dirs, regenerate):
    outdir = results_dirs[0]
    print(f"Writing analysis to {outdir}")

    df = load_graph_planning_results(results_dirs, regenerate)

    def _shorten(c):
        return shorten(c.split('/')[0])[:16]

    df['x_name'] = df['classifier_name'].map(_shorten)

    hue = 'used_augmentation'

    _, ax = violinplot(df, outdir, hue, 'task_error', "Task Error")
    _, ymax = ax.get_ylim()
    ax.set_ylim([0, ymax])

    _, ax = violinplot(df, outdir, hue, 'normalized_model_error', 'Normalized Model Error')
    _, ymax = ax.get_ylim()
    ax.set_ylim([0, ymax])

    _, ax = violinplot(df, outdir, hue, 'combined_error', 'Combined Error')
    _, ymax = ax.get_ylim()
    ax.set_ylim([0, ymax])

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(
        ax=ax,
        data=df,
        x='used_augmentation',
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
    ax.set_title('success')
    plt.savefig(outdir / f'success.png')

    _, ax = violinplot(df, outdir, hue, 'min_planning_error', "Planning Error")

    _, ax = violinplot(df, outdir, hue, 'min_error_discrepancy', "Error Discrepancy")

    plt.show()


@ros_init.with_ros("analyse_graph_planning")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dirs', help='results directory', type=pathlib.Path, nargs='+')
    parser.add_argument('--regenerate', action='store_true')

    args = parser.parse_args()

    plt.style.use('slides')

    analyze_graph_planning(results_dirs=args.results_dirs, regenerate=args.regenerate)


if __name__ == '__main__':
    main()
