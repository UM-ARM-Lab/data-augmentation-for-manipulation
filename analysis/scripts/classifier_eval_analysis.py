#!/usr/bin/env python
import argparse

from dynamo_pandas import get_df

from link_bot_classifiers import dynamodb_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug')
    args = parser.parse_args()

    df = get_df(table=dynamodb_utils.table(args.debug))
    print(df)


if __name__ == '__main__':
    main()
