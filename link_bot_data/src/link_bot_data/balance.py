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
    # FIXME: redo this when I redo my dataset code
    positive_examples = dataset.filter(label_is(1))
    negative_examples = dataset.filter(label_is(0))

    # negative_examples = negative_examples.repeat()
    # print("UP-SAMPLING POSITIVE EXAMPLES!!!")
    # positive_examples = positive_examples.repeat()

    print("DOWN-SAMPLING TO BALANCE")
    balanced_dataset = tf.data.Dataset.zip((positive_examples.dataset, negative_examples.dataset))
    balanced_dataset = balanced_dataset.flat_map(flatten_concat_pairs)

    return balanced_dataset
