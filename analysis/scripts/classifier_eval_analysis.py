#!/usr/bin/env python
import argparse
import pathlib
from time import time

import tabulate
from dynamo_pandas import get_df

from analysis.analyze_results import load_table_specs, generate_tables
from link_bot_data import dynamodb_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tables_config', type=pathlib.Path)
    parser.add_argument('--debug')
    parser.add_argument('--latex')
    args = parser.parse_args()

    df = get_df(table=dynamodb_utils.classifier_table(args.debug))
    df = df.loc[df['mode'] == 'all']

    if args.latex:
        table_format = 'latex_raw'
    else:
        table_format = tabulate.simple_separated_format("\t")

    table_specs = load_table_specs(args.tables_config, table_format)

    root = pathlib.Path("/media/shared/classifier_eval")
    outdir = root / f'{int(time())}'
    outdir.mkdir(parents=True)

    generate_tables(df, outdir, table_specs)


if __name__ == '__main__':
    main()
