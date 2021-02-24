import tensorflow as tf

from link_bot_data.base_dataset import SizedTFDataset


def label_is(label_is, key='is_close'):
    def __filter(example):
        result = tf.squeeze(tf.equal(example[key][1], label_is))
        return result

    return __filter


def flatten_concat_pairs(ex_pos, ex_neg):
    flat_pair = tf.data.Dataset.from_tensors(ex_pos).concatenate(tf.data.Dataset.from_tensors(ex_neg))
    return flat_pair


def balance(dataset: SizedTFDataset):
    # # print("UP-SAMPLING POSITIVE EXAMPLES!!!")
    #
    # # print("DOWN-SAMPLING TO BALANCE")
    # balanced_dataset = tf.data.Dataset.zip((positive_examples.dataset, negative_examples.dataset))
    # balanced_dataset = balanced_dataset.flat_map(flatten_concat_pairs)

    balanced_dataset = SizedTFDataset.balance(positive_examples, negative_examples)
    return balanced_dataset
