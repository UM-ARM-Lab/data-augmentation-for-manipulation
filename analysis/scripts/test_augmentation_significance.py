#!/usr/bin/env python
import argparse

from colorama import Fore
from dynamo_pandas import get_df
import scipy.stats

from analysis.proxy_datasets import proxy_datasets_dict
from link_bot_data import dynamodb_utils
from link_bot_pycommon.pandas_utils import df_where


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug')
    args = parser.parse_args()

    df = get_df(table=dynamodb_utils.classifier_table(args.debug))

    df = filter_df_for_experiment(df)

    test_improvement_of_aug_on_car_for_metric(df, proxy_metric_name='ras')
    test_improvement_of_aug_on_car_for_metric(df, proxy_metric_name='ncs')
    test_improvement_of_aug_on_car_for_metric(df, proxy_metric_name='hrs')


def test_improvement_significance_for_metric(df, proxy_dataset_name, groupby, proxy_metric_name):
    cld = '/media/shared/classifier_data/'
    proxy_dataset_path = cld + proxy_datasets_dict[proxy_dataset_name][proxy_metric_name]
    df_p = df_where(df, 'dataset_dirs', proxy_dataset_path)
    agg = {k: 'first' for k in groupby}
    metric_name = 'accuracy on negatives'
    agg[metric_name] = 'mean'
    agg['classifier'] = 'first'
    l = ['do_augmentation', 'on_invalid_aug', 'fine_tuning_take', 'fine_tuned_from', 'fine_tuning_dataset_dirs']

    no_ft_baseline_all = df_p.loc[
        (df_p['classifier_source_env'] == 'floating_boxes') & (df_p['do_augmentation'] == 0)]
    no_ft_baseline_all.set_index(groupby, inplace=True)
    no_ft_baseline_all_dropped = no_ft_baseline_all.droplevel(l)
    no_ft_baseline_all_dropped = no_ft_baseline_all_dropped[metric_name]
    offline_ft_aug_baseline_all = df_p.loc[
        (df_p['classifier_source_env'] == 'floating_boxes') & (df_p['do_augmentation'] == 1)]
    offline_ft_aug_baseline_all.set_index(groupby, inplace=True)
    offline_ft_aug_baseline_all_dropped = offline_ft_aug_baseline_all.droplevel(l)
    offline_ft_aug_baseline_all_dropped = offline_ft_aug_baseline_all_dropped[metric_name]
    improvement = offline_ft_aug_baseline_all_dropped - no_ft_baseline_all_dropped

    print(Fore.CYAN + f"All Results {proxy_dataset_path}, {metric_name}" + Fore.RESET)
    drop_for_display = [
        'classifier_source_env',
        'dataset_dirs',
        'balance',
        'mode'
    ]
    print(improvement.round(3).droplevel(drop_for_display))
    print('MEAN:', improvement.mean())

    p = scipy.stats.ttest_1samp(improvement, 0).pvalue
    flag = '!' if p < 0.01 else ''
    print(Fore.CYAN + f'p-value for improvement {flag}{p:0.4f}' + Fore.RESET)
    print()


def test_improvement_of_aug_on_car_for_metric(df, proxy_metric_name):
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
    test_improvement_significance_for_metric(df, proxy_dataset_name, groupby, proxy_metric_name)


def filter_df_for_experiment(df):
    experiment_type = 'offline_v3'
    df = df.loc[df['mode'] == 'all']
    print(experiment_type)
    if experiment_type == 'none':
        return df
    elif experiment_type == 'offline_v2':
        offline_ft_dataset = '/media/shared/classifier_data/val_floating_boxes_1622170084+fix-op'
        v2_car_aug = (df['classifier'].str.contains('fb2car_pre_aug') & df['fine_tuning_take'].isna() & (
                df['fine_tuning_dataset_dirs'] == offline_ft_dataset))
        no_aug = df['classifier'].str.contains('val_floating_boxes*')
        car_baseline = df['classifier'].str.contains('val_car_new*')
        df = df.loc[v2_car_aug | no_aug | car_baseline]
    elif experiment_type == 'offline_v3':
        offline_ft_dataset = '/media/shared/classifier_data/val_floating_boxes_1622170084+fix-op'
        v3_car_aug = (df['classifier'].str.contains('fb2car_v3_drop') & df['fine_tuning_take'].isna() & (
                df['fine_tuning_dataset_dirs'] == offline_ft_dataset))
        no_aug = (df['classifier'].str.contains('val_floating_boxes1') |
                  df['classifier'].str.contains('val_floating_boxes2') |
                  df['classifier'].str.contains('val_floating_boxes3') |
                  df['classifier'].str.contains('val_floating_boxes4'))
        car_baseline = df['classifier'].str.contains('val_car_new*')
        df = df.loc[v3_car_aug | no_aug | car_baseline]
    elif experiment_type == 'online':
        ft_dataset = '/media/shared/classifier_data/val_car_feasible_1614981888+op2'
        cond1 = (df['fine_tuning_dataset_dirs'] == ft_dataset)
        cond2 = df['fine_tuning_dataset_dirs'].isna()
        df = df.loc[cond1 | cond2]
        cond1 = (df['fine_tuning_take'] == 500)
        cond2 = (df['fine_tuning_take'].isna() & (~df['do_augmentation']))
        df = df.loc[cond1 | cond2]
    elif experiment_type == 'take10':
        ft_dataset = '/media/shared/classifier_data/val_car_feasible_1614981888+op2'
        cond1 = (df['fine_tuning_dataset_dirs'] == ft_dataset)
        cond2 = df['fine_tuning_dataset_dirs'].isna()
        df = df.loc[cond1 | cond2]
        drop_indices = df.index[(df['fine_tuning_take'] != 10) & df['do_augmentation']]
        df.drop(drop_indices, inplace=True)
    elif experiment_type == 'full':
        ft_dataset = '/media/shared/classifier_data/val_car_feasible_1614981888+op2'
        cond1 = (df['fine_tuning_dataset_dirs'] == ft_dataset)
        cond2 = df['fine_tuning_dataset_dirs'].isna()
        df = df.loc[cond1 | cond2]
        df = df.loc[df['fine_tuning_take'].isna()]
    return df


if __name__ == '__main__':
    main()
