#!/usr/bin/env python
import argparse
import functools
import pathlib
from operator import ior, iand

import pandas as pd
import scipy.stats
from colorama import Fore
from dynamo_pandas import get_df

from analysis.proxy_datasets import proxy_datasets_dict
from link_bot_data import dynamodb_utils
from link_bot_pycommon.pandas_utils import df_where

proxy_dataset_name = 'car1'
groupby = [
    "do_augmentation",
    'fine_tuned_from',
    "fine_tuning_take",
    "classifier_source_env",
    "dataset_dirs",
    "mode",
    'original_training_seed',
    # "balance",
    "fine_tuning_dataset_dirs",
    'on_invalid_aug',
]

cld = '/media/shared/classifier_data/'


def main():
    pd.options.display.max_colwidth = 100
    parser = argparse.ArgumentParser()
    parser.add_argument('contains', type=str, help="includes classifiers with 'contains' in their name")
    parser.add_argument('--debug')
    args = parser.parse_args()

    df = get_df(table=dynamodb_utils.classifier_table(args.debug))

    df = filter_df_for_experiment(df, args.contains)

    print("Classifiers:")
    proxy_dataset_path = cld + proxy_datasets_dict[proxy_dataset_name]['ras']
    df_p = df_where(df, 'dataset_dirs', proxy_dataset_path)
    print('\n'.join(df_p['classifier'].sort_values().values))
    print()

    test_improvement_of_aug_on_car_for_metric(df, proxy_metric_name='ras')
    test_improvement_of_aug_on_car_for_metric(df, proxy_metric_name='ncs')
    test_improvement_of_aug_on_car_for_metric(df, proxy_metric_name='hrs')


def test_improvement_of_aug_on_car_for_metric(df, proxy_metric_name):
    proxy_dataset_path = cld + proxy_datasets_dict[proxy_dataset_name][proxy_metric_name]
    df_p = df_where(df, 'dataset_dirs', proxy_dataset_path)

    metric_name = 'accuracy on negatives'
    # drop things which are the thing we expect to differ between baseline and our method?
    l = [
        'do_augmentation',
        'on_invalid_aug',
        'fine_tuning_take',
        'fine_tuned_from',
        'fine_tuning_dataset_dirs',
        # 'class_balance',
    ]

    no_aug_baseline_all = df_p.loc[(df_p['classifier_source_env'] == 'floating_boxes') & (df_p['do_augmentation'] == 0)]
    no_aug_baseline_all.set_index(groupby, inplace=True)
    no_aug_baseline_all_dropped = no_aug_baseline_all.droplevel(l)
    no_aug_baseline_all_dropped = no_aug_baseline_all_dropped[metric_name]
    aug_baseline_all = df_p.loc[(df_p['classifier_source_env'] == 'floating_boxes') & (df_p['do_augmentation'] == 1)]
    aug_baseline_all.set_index(groupby, inplace=True)
    aug_baseline_all_dropped = aug_baseline_all.droplevel(l)
    aug_baseline_all_dropped = aug_baseline_all_dropped[metric_name]

    car_baseline_all = df_p.loc[df_p['classifier'].str.contains('val_car_new') & (df_p['do_augmentation'] == 0)]
    car_baseline_all.set_index(groupby, inplace=True)
    car_baseline_all_dropped = car_baseline_all.droplevel(l)
    car_baseline_all_dropped = car_baseline_all_dropped[metric_name]

    improvement = aug_baseline_all_dropped - no_aug_baseline_all_dropped
    possible_improvement = car_baseline_all_dropped.mean() - no_aug_baseline_all_dropped.mean()

    print(Fore.CYAN + f"All Results {proxy_dataset_path}, {metric_name}" + Fore.RESET)

    def print_stats(values, name):
        drop_for_display = [
            'classifier_source_env',
            'dataset_dirs',
            # 'balance',
            'mode'
        ]
        print(values.round(3).droplevel(drop_for_display).reset_index())
        stats_and_formats = [
            (values.mean(), '{:.4f}'),
            (values.std(), '{:.4f}'),
            (values.count(), '{:d}'),
        ]
        stats_formatted = [fmt.format(v) for v, fmt in stats_and_formats]
        print(Fore.GREEN + f"{name}: " + ' '.join(stats_formatted) + Fore.RESET)

    print_stats(improvement, "improvement")
    print(f"estimated possible improvement {possible_improvement:.4f}")

    p = scipy.stats.ttest_1samp(improvement, 0).pvalue
    flag = '!' if p < 0.01 else ''
    print(Fore.CYAN + f'p-value for improvement {flag}{p:0.4f}' + Fore.RESET)
    print()


def filter_df_for_experiment(df, classifier_contains: str):
    df = df.loc[df['mode'] == 'all']
    offline_ft_dataset = '/media/shared/classifier_data/val_floating_boxes_1622170084+fix-op'
    aug_cond = functools.reduce(iand, [
        df['classifier'].str.contains(classifier_contains),
        (df['fine_tuning_dataset_dirs'] == offline_ft_dataset),
    ])
    aug_df = df.loc[aug_cond]
    comparable_no_aug_classifiers = aug_df['fine_tuned_from'].map(lambda c: pathlib.Path(c).parts[4]).unique().tolist()
    comparable_no_aug_classifiers_cond = [df['classifier'].str.contains(c) for c in comparable_no_aug_classifiers]
    no_aug = functools.reduce(ior, comparable_no_aug_classifiers_cond)
    car_baseline = df['classifier'].str.contains('val_car_new*')
    df = df.loc[aug_cond | no_aug | car_baseline]
    return df


if __name__ == '__main__':
    main()
