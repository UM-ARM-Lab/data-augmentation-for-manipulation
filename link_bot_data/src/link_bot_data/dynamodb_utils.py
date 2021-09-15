from typing import Callable

from colorama import Fore
from dynamo_pandas import get_df
from halo import halo

primary_key = 'uuid'


def classifier_table(debug=False):
    if debug:
        input(Fore.RED + "Press enter to confirm you want to use the debugging database" + Fore.RESET)
        return 'debugging'
    else:
        return 'classifier-evaluation'


def update_item(client, dtype, item, k, table, v):
    client.update_item(
        TableName=table,
        Key={
            'uuid': item['uuid'],
        },
        UpdateExpression=f"SET {k} = :v",
        ExpressionAttributeValues={
            ':v': {dtype: v},
        },
    )


def update_classifier_db(client, table, f: Callable):
    response = client.scan(TableName=table)
    while True:
        items = response["Items"]

        for item in items:
            result = f(item)
            if result is not None:
                v, dtype, k = result
                update_item(client, dtype, item, k, table, v)

        if "LastEvaluatedKey" not in response:
            break
        key = response["LastEvaluatedKey"]
        response = client.scan(TableName=table, ExclusiveStartKey=key)


@halo.Halo("getting classifier df")
def get_classifier_df(debug=False):
    df = get_df(table=classifier_table(debug))
    return df
