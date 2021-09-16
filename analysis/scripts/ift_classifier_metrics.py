#!/usr/bin/env python
import argparse
import functools
import operator
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.proxy_datasets import proxy_datasets_dict
from analysis.results_figures import lineplot
from analysis.results_utils import classifier_name_to_iter
from link_bot_data.dynamodb_utils import get_classifier_df
from link_bot_pycommon.pandas_utils import rlast
from moonshine.filepath_tools import load_hjson


def main():
    pd.options.display.max_rows = 999

    parser = argparse.ArgumentParser()
    parser.add_argument('results_dirs', type=pathlib.Path, nargs='+')

    args = parser.parse_args()

    plt.style.use('slides')

    df = get_classifier_df()

    classifier_names = []
    is_iter_cl_conds = []
    for results_dir in args.results_dirs:
        results_name = results_dir.name
        pparams = load_hjson(results_dir / 'planning_results' / 'iteration_0001_planning' / 'metadata.hjson')
        checkpoint_type = pathlib.Path(pparams['planner_params']['classifier_model_dir'][0]).parts[-1]
        classifier_name = f"{results_name}.*{checkpoint_type}"
        classifier_names.append(classifier_name)
        is_iter_cl = df['classifier'].str.contains(classifier_name)
        is_iter_cl_conds.append(is_iter_cl)

    is_iter_cl_any = functools.reduce(operator.ior, is_iter_cl_conds)
    is_first_iter_cl = df['classifier'].str.contains('untrained-1')
    z = df.loc[is_iter_cl_any | is_first_iter_cl]

    outdir = args.results_dirs[0]

    plot_proxy_dataset_metric(z, 'hrs', 'accuracy on negatives', f'Specificity on HRS')
    plt.savefig(outdir / 'spec_hrs.png')
    plot_proxy_dataset_metric(z, 'ncs', 'accuracy on negatives', f'Specificity on NCS')
    plt.savefig(outdir / 'spec_ncs.png')
    plot_proxy_dataset_metric(z, 'ras', 'accuracy', f'Accuracy on RAS')
    plt.savefig(outdir / 'acc_ras.png')

    # plot_mistakes_over_time(args.results_dir)
    # plt.savefig(outdir / 'mistakes.png')

    plt.show()


def plot_proxy_dataset_metric(df, proxy_dataset_type: str, metric_name: str, title: str):
    proxy_dataset_name = 'car1'
    proxy_dataset_path = proxy_datasets_dict[proxy_dataset_name][proxy_dataset_type]
    df = df.loc[df['dataset_dirs'] == proxy_dataset_path]
    # without this copy, there is a chained indexing warning
    df = df.copy()
    iter_key = 'ift_iteration'
    iter_value = df['classifier'].map(classifier_name_to_iter)
    df[iter_key] = iter_value

    fig, ax = lineplot(df, iter_key, metric_name, title, figsize=(10, 7), hue='full_retrain', style='do_augmentation')
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.01, 1.01)

    # NOTE: something is wrong with the rolling calculations here I think... some lines don't start at 9
    agg = {
        'full_retrain':      'mean',
        'do_augmentation': rlast,
        metric_name:         'mean',
        iter_key:            rlast,
    }
    df = df.loc[~df['full_retrain'].isna()]
    z = df.loc[(df['full_retrain'] == 1.0) & (df['do_augmentation'] == 0.0)]
    h=z.sort_values(iter_key).rolling(10).agg(agg)
    df_r = df.sort_values(iter_key).groupby(['full_retrain', 'do_augmentation']).rolling(10).agg(agg)
    z = df.sort_values(iter_key).groupby(['full_retrain', 'do_augmentation']).first()

    # fig, ax = lineplot(df_r, iter_key, metric_name, title + ' (rolling)', figsize=(10, 7), hue='full_retrain')
    # ax.set_xlim(0, 100)
    # ax.set_ylim(-0.01, 1.01)
    #
    # fig, ax = lineplot(df_r, iter_key, metric_name, title + ' (rolling)', figsize=(10, 7), hue='do_augmentation')
    # ax.set_xlim(0, 100)
    # ax.set_ylim(-0.01, 1.01)
    #
    # fig, ax = lineplot(df_r, iter_key, metric_name, title + ' (rolling)', figsize=(10, 7), hue='full_retrain', style='do_augmentation')
    # ax.set_xlim(0, 100)
    # ax.set_ylim(-0.01, 1.01)


def plot_mistakes_over_time(results_dir):
    mistakes_over_time_filename = results_dir / 'mistakes_over_time.hjson'
    mistakes_over_time = load_hjson(mistakes_over_time_filename)
    mistakes_over_time = {int(k): v for k, v in mistakes_over_time.items()}
    plt.figure()
    cumsum = [0] * len(mistakes_over_time[0].keys())
    for start_i, mistakes_over_time_i in sorted(mistakes_over_time.items()):
        start_i = int(start_i)
        iterations_i = [int(i) for i in mistakes_over_time_i.keys()]
        mistakes_i = [int(m) for m in mistakes_over_time_i.values()]
        mistakes_i_cumsum = np.array(cumsum[start_i:]) + mistakes_i

        for ii, m in zip(iterations_i, mistakes_i):
            cumsum[ii] += m
        plt.plot(iterations_i, mistakes_i_cumsum, label=f'iter={start_i}')
    plt.xlabel("iteration")
    plt.ylabel("num mistakes")
    plt.title("mistakes over time")


if __name__ == '__main__':
    main()
