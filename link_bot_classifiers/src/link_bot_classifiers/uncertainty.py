import tensorflow as tf


def make_max_class_prob(probabilities):
    other_class_probabilities = 1 - probabilities
    return tf.maximum(probabilities, other_class_probabilities)
