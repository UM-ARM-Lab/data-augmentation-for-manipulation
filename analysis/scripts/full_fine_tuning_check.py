#!/usr/bin/env python
import argparse

import pandas as pd
import tabulate
from dynamo_pandas import get_df

from analysis.analyze_results import generate_tables, make_table_specs
from analysis.results_utils import try_load_classifier_params
from link_bot_data import dynamodb_utils
from link_bot_data.dynamodb_utils import get_classifier_df
from link_bot_pycommon.pandas_utils import df_where


def make_tables_specs(table_format: str):
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
        {
            'type':       'MyTable',
            'name':       'mean&std',
            'header':     [
                'Classifier Source Env',
                'N',
                'FT Conv',
                'FT LSTM',
                'FT Dense',
                'FT Output',
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
                [[groupby, "classifier_source_env", "first"]],
                [[groupby, "classifier_source_env", "count"]],
                [[groupby, "fine_tune_conv", "first"]],
                [[groupby, "fine_tune_lstm", "first"]],
                [[groupby, "fine_tune_dense", "first"]],
                [[groupby, "fine_tune_output", "first"]],
                [[groupby, 'accuracy', ['mean', 'std']]],
                [[groupby, 'precision', ['mean', 'std']]],
                [[groupby, 'accuracy on positives', ['mean', 'std']]],
                [[groupby, 'accuracy on negatives', ['mean', 'std']]],
                [[groupby, 'loss', ['mean', 'std']]],
            ],
        },
        {
            'type':       'PValuesTable',
            'name':       'Accuracy',
            'reductions': [
                [[groupby, 'accuracy', None]],
            ],

        },
        {
            'type':       'PValuesTable',
            'name':       'Precision',
            'reductions': [
                [[groupby, 'precision', None]],
            ],

        },
        {
            'type':       'PValuesTable',
            'name':       'Accuracy',
            'reductions': [
                [[groupby, 'accuracy on positives', None]],
            ],

        },
        {
            'type':       'PValuesTable',
            'name':       'AoN',
            'reductions': [
                [[groupby, 'accuracy on negatives', None]],
            ],

        },
        {
            'type':       'PValuesTable',
            'name':       'loss',
            'reductions': [
                [[groupby, 'loss', None]],
            ],

        },
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

    df = get_classifier_df(args.debug)

    df = filter_df_for_experiment(df)

    s = make_tables_specs(table_format)
    generate_tables(df=df, outdir=None, table_specs=s)


def filter_df_for_experiment(df):
    df = df_where(df, 'dataset_dirs', '/media/shared/classifier_data/val_car_feasible_1614981888+op2')
    og_cond1 = df['classifier'].str.contains('val_car_new')
    og_cond2 = df['fine_tuning_dataset_dirs'].isna()
    original_car_classifier = df.loc[og_cond1 & og_cond2]
    full_fine_tuned_classifier = df.loc[
        df['fine_tuning_take'].isna() &
        (df['fine_tuning_dataset_dirs'] == '/media/shared/classifier_data/val_car_feasible_1614981888+op2') &
        (df['do_augmentation'] == 0) &
        (df['learning_rate'] == 0.0001)
        ]
    df = pd.concat([original_car_classifier, full_fine_tuned_classifier], axis=0)
    return df


if __name__ == '__main__':
    main()
