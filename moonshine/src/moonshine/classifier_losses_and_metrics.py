from typing import Dict

import tensorflow as tf
from colorama import Fore
from tensorflow.keras.metrics import *


def class_weighted_binary_classification_sequence_loss_function(dataset_element, predictions, key='is_close'):
    # skip the first element, the label will always be 1
    is_close = dataset_element[key][:, 1:]
    labels = tf.expand_dims(is_close, axis=2)
    logits = predictions['logits']
    bce = tf.keras.losses.binary_crossentropy(y_true=labels, y_pred=logits, from_logits=True)
    total_bce = class_weighted_mean_loss(bce, is_close)
    return total_bce


def class_weighted_mean_loss(bce, positives):
    # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#class_weights
    negatives = 1 - positives
    n_positive = tf.math.reduce_sum(positives)
    n_negative = tf.math.reduce_sum(negatives)
    n_total = n_positive + n_negative
    if tf.logical_or(tf.equal(n_positive, 0), tf.equal(n_negative, 0)):
        weight_for_negative = tf.constant(1, dtype=tf.float32)
        weight_for_positive = tf.constant(1, dtype=tf.float32)
    else:
        weight_for_negative = n_total / 2.0 / n_negative
        weight_for_positive = n_total / 2.0 / n_positive
    # tf.print(f"\n{n_positive:.3f}, {n_negative:.3f}, {weight_for_positive:.3f}, {weight_for_negative:.3f}")
    weighted_bce = tf.math.add(tf.math.multiply(bce, positives * weight_for_positive),
                               tf.math.multiply(bce, negatives * weight_for_negative))
    if tf.equal(n_negative, 0):
        weight_for_negative = tf.constant(1, dtype=tf.float32)
        weight_for_positive = tf.constant(1, dtype=tf.float32)
    # mean over batch & time
    total_bce = tf.reduce_mean(weighted_bce)
    return total_bce


def binary_classification_loss_function(dataset_element, predictions):
    label = dataset_element['label']
    logit = predictions['logits'][:, -1]
    bce = tf.keras.losses.binary_crossentropy(y_true=label, y_pred=logit, from_logits=True)
    # mean over batch & time
    total_bce = tf.reduce_mean(bce)
    return total_bce


def binary_classification_metrics_function(dataset_element, predictions):
    label = dataset_element['label']
    logit = predictions['logits'][:, -1]
    accuracy_over_time = tf.keras.metrics.binary_accuracy(y_true=label, y_pred=logit)
    average_accuracy = tf.reduce_mean(accuracy_over_time)
    return {
        'accuracy': average_accuracy
    }


def mdn_sequence_likelihood(dataset_element, predictions):
    del dataset_element  # unused
    valid_log_likelihood = predictions['valid_log_likelihood']
    return -tf.reduce_mean(valid_log_likelihood)
