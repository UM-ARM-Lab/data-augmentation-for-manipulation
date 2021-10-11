import os

from dynamo_pandas.transactions import put_item
import pickle

from link_bot_data import dynamodb_utils
from link_bot_data.dynamodb_utils import get_classifier_df


def main():
    # os.environ['AWS_PROFILE'] = 'personal'
    # df = get_classifier_df()
    # with open("tmp.pkl", 'wb') as f:
    #     pickle.dump(df, f)
    with open("tmp.pkl", 'rb') as f:
        df = pickle.load(f)

    for item in df.to_dict(orient='records'):
        put_item(item=item, table=dynamodb_utils.classifier_table())


if __name__ == '__main__':
    main()
