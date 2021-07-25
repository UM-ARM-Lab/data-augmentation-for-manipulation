#!/usr/bin/env python
import scipy.stats
import argparse

from colorama import Fore
from dynamo_pandas import get_df

from analysis.proxy_datasets import proxy_datasets_dict
from link_bot_data import dynamodb_utils
from link_bot_pycommon.pandas_utils import df_where


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug')
    parser.add_argument('--latex')
    args = parser.parse_args()

    df = get_df(table=dynamodb_utils.classifier_table(args.debug))

    df = filter_df_for_experiment(df)

    test_drop_vs_original_for_metric(df, proxy_metric_name='ras')
    test_drop_vs_original_for_metric(df, proxy_metric_name='ncs')
    test_drop_vs_original_for_metric(df, proxy_metric_name='hrs')


def test_drop_vs_original_for_metric(df, proxy_metric_name):
    proxy_dataset_name = 'car1'
    groupby = [
        "do_augmentation",
        'fine_tuned_from',
        "fine_tuning_take",
        "classifier_source_env",
        "dataset_dirs",
        "mode",
        'original_training_seed',
        "balance",
        "fine_tuning_dataset_dirs",
        'on_invalid_aug',
    ]
    test_significance_for_metric(df, proxy_dataset_name, groupby, proxy_metric_name)


def test_significance_for_metric(df, proxy_dataset_name, groupby, proxy_metric_name):
    cld = '/media/shared/classifier_data/'
    df_p = df_where(df, 'dataset_dirs', cld + proxy_datasets_dict[proxy_dataset_name][proxy_metric_name])
    metric_name = 'accuracy on negatives'
    reduced = df_p.groupby(groupby, dropna=False).agg({metric_name: 'mean'})
    drop = reduced.query("on_invalid_aug == 'drop'").droplevel("on_invalid_aug")
    original = reduced.query("on_invalid_aug == 'original'").droplevel("on_invalid_aug")
    delta = drop - original

    print(Fore.CYAN + "All Results" + Fore.RESET)
    drop_for_display = [
        'fine_tuning_take',
        'do_augmentation',
        'dataset_dirs',
        'fine_tuning_dataset_dirs',
        'balance',
        'classifier_source_env',
        'original_training_seed',
        'mode'
    ]
    print(delta.round(3).droplevel(drop_for_display))

    p = scipy.stats.ttest_1samp(delta.squeeze(), 0).pvalue
    flag = '!' if p < 0.01 else ''
    print(Fore.GREEN + f'p-value for improvement {flag}{p:0.4f}' + Fore.RESET)


def filter_df_for_experiment(df):
    df = df.loc[df['mode'] == 'all']
    df = df.loc[df['classifier'].str.contains('fb2car_v3')]
    return df


if __name__ == '__main__':
    main()
