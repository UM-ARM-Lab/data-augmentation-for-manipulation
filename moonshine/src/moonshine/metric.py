import tensorflow as tf


class Metric:

    @staticmethod
    def is_better_than(a, b):
        raise NotImplementedError()

    @staticmethod
    def key():
        raise NotImplementedError()

    @staticmethod
    def worst():
        raise NotImplementedError()


class LossMetric(Metric):

    @staticmethod
    def is_better_than(a, b):
        return a < b

    @staticmethod
    def key():
        return "loss"

    @staticmethod
    def worst():
        return 1000


class AccuracyMetric(Metric):

    @staticmethod
    def is_better_than(a, b):
        if b is None:
            return True
        return a > b

    @staticmethod
    def key():
        return "accuracy"

    @staticmethod
    def worst():
        return 0


# TODO make tests for these


def fp(y_true, y_pred, threshold=0.5):
    return tf.cast(tf.math.count_nonzero((1 - y_true) * tf.cast(y_pred > threshold, tf.float32)), tf.float32)


def tn(y_true, y_pred, threshold=0.5):
    return tf.cast(tf.math.count_nonzero((1 - y_true) * tf.cast(y_pred <= threshold, tf.float32)), tf.float32)


def fn(y_true, y_pred, threshold=0.5):
    return tf.cast(tf.math.count_nonzero(y_true * tf.cast(y_pred <= threshold, tf.float32)), tf.float32)


def tp(y_true, y_pred, threshold=0.5):
    return tf.cast(tf.math.count_nonzero(y_true * tf.cast(y_pred > threshold, tf.float32)), tf.float32)


def accuracy_on_negatives(y_true, y_pred, threshold=0.5):
    true_negatives = tn(y_true, y_pred, threshold=threshold)
    false_positives = fp(y_true, y_pred, threshold=threshold)
    accuracy = tf.math.divide_no_nan(true_negatives, true_negatives + false_positives)
    return accuracy


def recall(y_true, y_pred, threshold=0.5):
    true_positives = tp(y_true, y_pred, threshold=threshold)
    false_negatives = fn(y_true, y_pred, threshold=threshold)
    return tf.math.divide_no_nan(true_positives, true_positives + false_negatives)


def precision(y_true, y_pred, threshold=0.5):
    true_positives = tp(y_true, y_pred, threshold=threshold)
    false_positives = fp(y_true, y_pred, threshold=threshold)
    return tf.math.divide_no_nan(true_positives, true_positives + false_positives)


def mistakes(y_true, y_pred):
    y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
    return tf.cast(tf.math.count_nonzero(y_true != y_pred_binary), tf.float32)


def fp_rate(y_true, y_pred):
    false_positives = fp(y_true, y_pred)
    m = mistakes(y_true, y_pred)
    return tf.math.divide_no_nan(false_positives, m)


def fn_rate(y_true, y_pred):
    false_negatives = fn(y_true, y_pred)
    m = mistakes(y_true, y_pred)
    return tf.math.divide_no_nan(false_negatives, m)
