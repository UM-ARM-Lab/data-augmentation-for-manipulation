import tensorflow as tf
from tensorflow.keras.metrics import Metric, FalsePositives, FalseNegatives


class CheckpointMetric:

    @staticmethod
    def is_better_than(a, b):
        raise NotImplementedError()

    @staticmethod
    def key():
        raise NotImplementedError()

    @staticmethod
    def worst():
        raise NotImplementedError()


class LossCheckpointMetric(CheckpointMetric):

    @staticmethod
    def is_better_than(a, b):
        return a < b

    @staticmethod
    def key():
        return "loss"

    @staticmethod
    def worst():
        return 1000


class AccuracyCheckpointMetric(CheckpointMetric):

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


class BinaryAccuracyOnPositives(Metric):
    def __init__(self):
        super().__init__()
        self.threshold = 0.5
        self.positive_count = self.add_weight(name='positive_count', shape=[], initializer='zeros')
        self.true_positives_count = self.add_weight(name='true_positives_count', shape=[], initializer='zeros')

    def result(self):
        return self.true_positives_count / self.positive_count

    def update_state(self, y_true, y_pred):
        true_is_positive = tf.cast(y_true, tf.bool)
        pred_is_positive = (y_pred > self.threshold)
        is_tp = tf.logical_and(pred_is_positive, true_is_positive)
        tp_count = tf.reduce_sum(tf.cast(is_tp, tf.float32))
        self.true_positives_count.assign_add(tp_count)

        positive_count = tf.reduce_sum(tf.cast(true_is_positive, tf.float32))
        self.positive_count.assign_add(positive_count)

    def reset_states(self):
        super().reset_states()
        self.positive_count.assign(0)
        self.true_positives_count.assign(0)


class BinaryAccuracyOnNegatives(Metric):
    def __init__(self):
        super().__init__()
        self.threshold = 0.5
        self.negative_count = self.add_weight(name='negative_count', shape=[], initializer='zeros')
        self.true_negative_count = self.add_weight(name='true_negative_count', shape=[], initializer='zeros')

    def result(self):
        return self.true_negative_count / self.negative_count

    def update_state(self, y_true, y_pred):
        true_is_negative = tf.logical_not(tf.cast(y_true, tf.bool))
        pred_is_negative = tf.logical_not(y_pred > self.threshold)
        is_tn = tf.logical_and(pred_is_negative, true_is_negative)
        tn_count = tf.reduce_sum(tf.cast(is_tn, tf.float32))
        self.true_negative_count.assign_add(tn_count)

        negative_count = tf.reduce_sum(tf.cast(true_is_negative, tf.float32))
        self.negative_count.assign_add(negative_count)

    def reset_states(self):
        super().reset_states()
        self.negative_count.assign(0)
        self.true_negative_count.assign(0)


class LossMetric(Metric):
    """ just takes the average assuming it's a scalar """

    def __init__(self):
        super().__init__()
        self.sum = self.add_weight(name='sum', shape=[], initializer='zeros')
        self.count = self.add_weight(name='count', shape=[], initializer='zeros')

    def result(self):
        return self.sum / self.count

    def update_state(self, loss):
        self.sum.assign_add(tf.reduce_sum(loss))
        self.count.assign_add(1)

    def reset_states(self):
        super().reset_states()
        self.sum.assign(0)
        self.count.assign(0)


class FalsePositiveMistakeRate(FalsePositives):
    """ Ratio of FP to total Mistakes """

    def __init__(self):
        super().__init__()
        self.mistakes_count = self.add_weight(name='mistakes', initializer='zeros')

    def result(self):
        return super().result() / self.mistakes_count

    def update_state(self, y_true, y_pred, **kwargs):
        super().update_state(y_true, y_pred, **kwargs)
        y_pred_binary = tf.cast(y_pred > self.thresholds, tf.float32)
        mistakes = tf.cast(tf.logical_not(tf.equal(y_true, y_pred_binary)), tf.float32)
        self.mistakes_count.assign_add(tf.reduce_sum(mistakes))

    def reset_states(self):
        super().reset_states()
        self.mistakes_count.assign(0)


class FalseNegativeMistakeRate(FalseNegatives):
    """ Ratio of FP to total Mistakes """

    def __init__(self):
        super().__init__()
        self.mistakes_count = self.add_weight(name='mistakes', initializer='zeros')

    def result(self):
        return super().result() / self.mistakes_count

    def update_state(self, y_true, y_pred, **kwargs):
        super().update_state(y_true, y_pred, **kwargs)
        y_pred_binary = tf.cast(y_pred > self.thresholds, tf.float32)
        mistakes = tf.cast(tf.logical_not(tf.equal(y_true, y_pred_binary)), tf.float32)
        self.mistakes_count.assign_add(tf.reduce_sum(mistakes))

    def reset_states(self):
        super().reset_states()
        self.mistakes_count.assign(0)

class FalsePositiveOverallRate(FalsePositives):
    """ Ratio of FP to total Overalls """

    def __init__(self):
        super().__init__()
        self.count = self.add_weight(name='mistakes', initializer='zeros')

    def result(self):
        return super().result() / self.count

    def update_state(self, y_true, y_pred, **kwargs):
        super().update_state(y_true, y_pred, **kwargs)
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def reset_states(self):
        super().reset_states()
        self.count.assign(0)


class FalseNegativeOverallRate(FalseNegatives):
    """ Ratio of FP to total Overalls """

    def __init__(self):
        super().__init__()
        self.count = self.add_weight(name='mistakes', initializer='zeros')

    def result(self):
        return super().result() / self.count

    def update_state(self, y_true, y_pred, **kwargs):
        super().update_state(y_true, y_pred, **kwargs)
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def reset_states(self):
        super().reset_states()
        self.count.assign(0)
