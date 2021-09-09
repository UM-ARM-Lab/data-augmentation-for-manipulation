#!/usr/bin/env python
import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dynamo_pandas import get_df

from analysis.proxy_datasets import proxy_datasets_dict
from analysis.results_figures import lineplot
from analysis.results_utils import classifier_name_to_iter
from arc_utilities import ros_init
from link_bot_data import dynamodb_utils
from moonshine.filepath_tools import load_hjson
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(0.1)


@ros_init.with_ros("ift_classifier_metrics")
def main():
    pd.options.display.max_rows = 999

    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', type=pathlib.Path)

    args = parser.parse_args()

    plt.style.use('slides')

    df = get_df(table=dynamodb_utils.classifier_table())
    is_iter_cl = df['classifier'].str.contains('untrained-aug-all-data.*latest_checkpoint')
    is_first_iter_cl = df['classifier'].str.contains('untrained-1')
    df = df.loc[is_iter_cl | is_first_iter_cl]

    plot_proxy_dataset_metric(df.copy(), 'hrs', 'accuracy on negatives', 'Specificity on HRS')
    plot_proxy_dataset_metric(df.copy(), 'ncs', 'accuracy on negatives', 'Specificity on NCS')
    plot_proxy_dataset_metric(df.copy(), 'ras', 'accuracy', 'Specificity on RAS')

    plot_mistakes_over_time(args.results_dir)
    plt.show()


def plot_proxy_dataset_metric(df, proxy_dataset_type: str, metric_name: str, title: str):
    proxy_dataset_name = 'car1'
    proxy_dataset_path = proxy_datasets_dict[proxy_dataset_name][proxy_dataset_type]
    df = df.loc[df['dataset_dirs'] == proxy_dataset_path]
    iter_key = 'ift_iteration'
    df[iter_key] = df['classifier'].map(classifier_name_to_iter)
    lineplot(df, iter_key, metric_name, title)
    return df, iter_key


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
