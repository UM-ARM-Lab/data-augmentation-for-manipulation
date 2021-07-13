#!/usr/bin/env python
from uuid import uuid4

from dynamo_pandas import get_df
from dynamo_pandas.transactions import put_item

from link_bot_classifiers import dynamodb_utils


def main():
    # client = boto3.client('dynamodb')

    print("BEFORE")
    df = get_df(table=dynamodb_utils.table_name)
    print(df)

    uuid = str(uuid4())
    item = {
        'uuid':        uuid,
        'mean':        0.1,
        'median':      0.2,
        'specificity': 0.3,
    }
    put_item(item=item, table=dynamodb_utils.table_name)

    print("AFTER")
    df = get_df(table=dynamodb_utils.table_name)
    print(df)


if __name__ == '__main__':
    main()
