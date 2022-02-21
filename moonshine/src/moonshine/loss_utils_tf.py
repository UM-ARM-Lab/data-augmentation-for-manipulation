import tensorflow as tf


def loss_on_dicts(loss_func, dict_true, dict_pred):
    loss_by_key = []
    for k, y_pred in dict_pred.items():
        y_true = dict_true[k]
        loss = loss_func(y_true=y_true, y_pred=y_pred)
        loss_by_key.append(loss)
    return tf.reduce_mean(loss_by_key)
