#!/usr/bin/env python
import argparse

import pandas as pd
import tabulate
from dynamo_pandas import get_df

from analysis.analyze_results import generate_tables, make_table_specs
from analysis.results_utils import try_load_classifier_params
from link_bot_data import dynamodb_utils
from link_bot_pycommon.pandas_utils import df_where


def make_tables_specs(table_format: str):
    def _table_config(_groupby, agg):
        return {
            'type':       'MyTable',
            'name':       'mean',
            'header':     [
                'Classifier Source Env',
                'N',
                'Dataset',
                'Fine-Tuning Take',
                'FT Conv',
                'FT LSTM',
                'FT Dense',
                'FT Output',
                'Learning Rate',
                'Accuracy',
                'Accuracy [std]',
                'Precision',
                'Precision [std]',
                'AoP',
                'AoP [std]',
                'AoN',
                'AoN [std]',
                'loss',
                'loss [std]',
            ],
            'reductions': [
                [[_groupby, "classifier_source_env", "first"]],
                [[_groupby, "classifier_source_env", "count"]],
                [[_groupby, "dataset_dirs", "first"]],
                [[_groupby, "fine_tuning_take", "first"]],
                [[_groupby, "fine_tune_conv", "first"]],
                [[_groupby, "fine_tune_lstm", "first"]],
                [[_groupby, "fine_tune_dense", "first"]],
                [[_groupby, "fine_tune_output", "first"]],
                [[_groupby, "learning_rate", "first"]],
                [[_groupby, 'accuracy', agg]],
                [[_groupby, 'precision', agg]],
                [[_groupby, 'accuracy on positives', agg]],
                [[_groupby, 'accuracy on negatives', agg]],
                [[_groupby, 'loss', agg]],
            ],
        }

    groupby = [
        "classifier_source_env",
        "dataset_dirs",
        "fine_tuning_dataset_dirs",
        "fine_tune_conv",
        "fine_tune_lstm",
        "fine_tune_dense",
        "fine_tune_output",
        "learning_rate",
    ]
    tables_config = [

        _table_config(groupby + ['uuid'], "mean"),
        _table_config(groupby, ["mean", "std"]),
    ]
    return make_table_specs(table_format, tables_config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug')
    parser.add_argument('--latex')
    args = parser.parse_args()

    if args.latex:
        table_format = 'latex_raw'
    else:
        table_format = tabulate.simple_separated_format("\t")

    df = get_df(table=dynamodb_utils.classifier_table(args.debug))

    df = filter_df_for_experiment(df)

    s = make_tables_specs(table_format)
    generate_tables(df=df, outdir=None, table_specs=s)

    for classifier_model_dir in df['classifier']:
        classifier_hparams = try_load_classifier_params(classifier_model_dir)
        print(classifier_hparams[''])


def filter_df_for_experiment(df):
    df = df_where(df, 'dataset_dirs', '/media/shared/classifier_data/val_car_feasible_1614981888+op2')
    og_cond1 = df['classifier'].str.contains('val_car_new')
    og_cond2 = df['fine_tuning_dataset_dirs'].isna()
    original_car_classifier = df.loc[og_cond1 & og_cond2]
    full_fine_tuned_classifier = df.loc[
        df['fine_tuning_take'].isna() &
        (df['fine_tuning_dataset_dirs'] == '/media/shared/classifier_data/val_car_feasible_1614981888+op2') &
        (df['do_augmentation'] == 0)
        ]
    df = pd.concat([original_car_classifier, full_fine_tuned_classifier], axis=0)
    return df


if __name__ == '__main__':
    main()
