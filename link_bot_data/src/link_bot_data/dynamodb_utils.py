from typing import Callable

from colorama import Fore

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
    for item in client.scan(TableName=table)['Items']:
        result = f(item)
        if result is not None:
            v, dtype, k = result
            update_item(client, dtype, item, k, table, v)