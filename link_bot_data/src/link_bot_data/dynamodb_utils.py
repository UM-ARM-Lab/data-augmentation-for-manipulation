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


def delete_item(client, table, uuid):
    client.delete_item(
        TableName=table,
        Key={
            'uuid': {'S': uuid},
        },
    )


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


def read_classifier_db_generator(client, table):
    response = client.scan(TableName=table)
    while True:
        items = response["Items"]

        for item in items:
            yield item

        if "LastEvaluatedKey" not in response:
            break
        key = response["LastEvaluatedKey"]
        response = client.scan(TableName=table, ExclusiveStartKey=key)


def update_classifier_db(client, table, f: Callable):
    for item in read_classifier_db_generator(client, table):
        result = f(item)
        if result is not None:
            v, dtype, k = result
            update_item(client, dtype, item, k, table, v)


def classifier_rows_are_close(item, item2):
    ignore_these_keys = ['uuid', 'time']
    for k, v1 in item.items():
        if k in ignore_these_keys:
            continue
        if k not in item2:
            return False

        v2 = item2[k]
        if 'NULL' in v1 and 'NULL' in v2:
            continue
        elif 'NULL' in v1:
            return False
        elif 'NULL' in v2:
            return False

        if 'S' in v1:
            v1 = v1['S']
            v2 = v2['S']
            if v1 != v2:
                return False
        elif 'N' in v1:
            v1 = float(v1['N'])
            v2 = float(v2['N'])
            if abs(v1 - v2) > 0.001:
                return False
        elif 'BOOL' in v1:
            v1 = bool(v1['BOOL'])
            v2 = bool(v2['BOOL'])
            if v1 != v2:
                return False

    return True


def remove_duplicates_in_classifier_db(client, table):
    for item in read_classifier_db_generator(client, table):
        uuid1 = item['uuid']['S']
        item.pop('uuid')
        for item2 in read_classifier_db_generator(client, table):
            # check if all but UUID match
            uuid2 = item2['uuid']['S']
            if uuid1 == uuid2:
                continue
            item2.pop('uuid')
            if classifier_rows_are_close(item, item2):
                print(uuid1, uuid2)
                delete_item(client, table, uuid2)


@halo.Halo("getting classifier df")
def get_classifier_df(debug=False):
    df = get_df(table=classifier_table(debug))
    return df
