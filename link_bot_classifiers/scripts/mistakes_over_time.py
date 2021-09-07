#!/usr/bin/env python

import argparse
import logging
import pathlib

import tensorflow as tf

from arc_utilities import ros_init
from link_bot_classifiers.train_test_classifier import ClassifierEvaluationFilter
from link_bot_pycommon.job_chunking import JobChunker


def is_mistake(example, predictions):
    is_close = example['is_close'][1:]
    probabilities = predictions['probabilities']

    is_predicted_close = tf.squeeze(probabilities > 0.5, axis=-1)
    is_close = tf.squeeze(tf.cast(is_close, tf.bool), axis=-1)

    return not tf.equal(is_predicted_close, is_close)


@ros_init.with_ros("mistakes_over_time")
def main():
    tf.get_logger().setLevel(logging.ERROR)
    parser = argparse.ArgumentParser()
    parser.add_argument('ift_dir', type=pathlib.Path)
    parser.add_argument('dataset_i', type=int)

    args = parser.parse_args()

    root = args.ift_dir

    dataset_dir = root / 'classifier_datasets' / f'iteration_{args.dataset_i:04d}_dataset'

    c = JobChunker(logfile_name=root / 'mistakes_over_time.hjson')
    sub = c.sub_chunker(str(args.dataset_i))

    mistakes = []
    for classifier_i in range(args.dataset_i, 200):
        checkpoint_dir = root / 'training_logdir' / f'iteration_{classifier_i:04d}_classifier_training_logdir'
        checkpoint = list(checkpoint_dir.iterdir())[0] / 'latest_checkpoint'

        key = str(classifier_i)
        count = sub.get_result(key)
        if count is None:
            evaluation = ClassifierEvaluationFilter(dataset_dirs=[dataset_dir],
                                                    checkpoint=checkpoint,
                                                    mode='all',
                                                    should_keep_example=is_mistake,
                                                    show_progressbar=False)

            count = len(list(evaluation))
            sub.store_result(key, count)
        mistakes.append(count)
        print(mistakes)


if __name__ == '__main__':
    main()
