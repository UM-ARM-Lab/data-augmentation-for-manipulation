#!/usr/bin/env python
import argparse
from functools import reduce
from operator import iand

from colorama import Fore
from dynamo_pandas import get_df
import scipy.stats

from analysis.proxy_datasets import proxy_datasets_dict
from link_bot_data import dynamodb_utils
from link_bot_pycommon.pandas_utils import df_where
import pandas as pd


def main():
    pd.options.display.max_colwidth = 100
    parser = argparse.ArgumentParser()
    parser.add_argument('contains', type=str, help="includes classifiers with 'contains' in their name")
    parser.add_argument('--debug')
    args = parser.parse_args()

    df = get_df(table=dynamodb_utils.classifier_table(args.debug))

    df = filter_df_for_experiment(df, args.contains)

    test_improvement_of_aug_on_car_for_metric(df, proxy_metric_name='ras')
    test_improvement_of_aug_on_car_for_metric(df, proxy_metric_name='ncs')
    test_improvement_of_aug_on_car_for_metric(df, proxy_metric_name='hrs')


def test_improvement_of_aug_on_car_for_metric(df, proxy_metric_name):
    proxy_dataset_name = 'car1'
    groupby = [
        "do_augmentation",
        'fine_tuned_from',
        "fine_tuning_take",
        "fine_tuning_seed",
        "classifier_source_env",
        "dataset_dirs",
        "mode",
        'original_training_seed',
        # "balance",
        "fine_tuning_dataset_dirs",
        'on_invalid_aug',
    ]

    cld = '/media/shared/classifier_data/'
    proxy_dataset_path = cld + proxy_datasets_dict[proxy_dataset_name][proxy_metric_name]
    df_p = df_where(df, 'dataset_dirs', proxy_dataset_path)

    print("Classifiers:")
    print(df_p['classifier'].sort_values())

    metric_name = 'accuracy on negatives'
    # drop things which are the thing we expect to differ between baseline and our method?
    l = ['do_augmentation', 'on_invalid_aug', 'fine_tuning_take', 'fine_tuning_dataset_dirs']

    no_aug_baseline_all = df_p.loc[(df_p['classifier_source_env'] == 'floating_boxes') & (df_p['do_augmentation'] == 0)]
    no_aug_baseline_all.set_index(groupby, inplace=True)
    no_aug_baseline_all_dropped = no_aug_baseline_all.droplevel(l)
    no_aug_baseline_all_dropped = no_aug_baseline_all_dropped[metric_name]
    aug_baseline_all = df_p.loc[(df_p['classifier_source_env'] == 'floating_boxes') & (df_p['do_augmentation'] == 1)]
    aug_baseline_all.set_index(groupby, inplace=True)
    aug_baseline_all_dropped = aug_baseline_all.droplevel(l)
    aug_baseline_all_dropped = aug_baseline_all_dropped[metric_name]
    improvement = aug_baseline_all_dropped - no_aug_baseline_all_dropped
    improvement_across_ft_seed = improvement.groupby('fine_tuned_from').agg("mean")

    print(Fore.CYAN + f"All Results {proxy_dataset_path}, {metric_name}" + Fore.RESET)
    print(improvement_across_ft_seed.round(3))
    print('MEAN:', improvement_across_ft_seed.mean())

    p = scipy.stats.ttest_1samp(improvement_across_ft_seed, 0).pvalue
    flag = '!' if p < 0.01 else ''
    print(Fore.CYAN + f'p-value for improvement {flag}{p:0.4f}' + Fore.RESET)
    print()


def filter_df_for_experiment(df, classifier_contains: str):
    df = df.loc[df['mode'] == 'all']
    online_ft_dataset = '/media/shared/classifier_data/val_car_feasible_1614981888+op2'
    cond = [
        df['classifier'].str.contains(classifier_contains),
        (df['fine_tuning_take'] == 100),
        (df['fine_tuning_dataset_dirs'] == online_ft_dataset),
    ]
    no_aug = ((df['classifier'].str.contains('fb2car_online100_baseline1')
               | df['classifier'].str.contains('fb2car_online100_baseline2')
               # | df['classifier'].str.contains('fb2car_online100_baseline3')
               # | df['classifier'].str.contains('fb2car_online100_baseline4')
               )
              & (df['do_augmentation'] == 0.0) & (df['fine_tuning_take'] == 100)
              & (df['fine_tuning_dataset_dirs'] == online_ft_dataset))

    df = df.loc[reduce(iand, cond) | no_aug]
    return df


if __name__ == '__main__':
    main()
