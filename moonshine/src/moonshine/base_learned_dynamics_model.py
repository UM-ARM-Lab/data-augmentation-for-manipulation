import tensorflow as tf

from moonshine.loss_utils_tf import loss_on_dicts


def dynamics_loss_function(example, predictions):
    loss = tf.keras.losses.MeanSquaredError()
    return loss_on_dicts(loss, dict_true=example, dict_pred=predictions)


def dynamics_points_metrics_function(example, predictions):
    metrics = dynamics_points_metrics(example, predictions)
    return metrics


def dynamics_points_metrics(example, predictions):
    metrics = {}
    for state_key, pred_state in predictions.items():
        true_state = example[state_key]
        pred_points = tf.reshape(pred_state, [pred_state.shape[0], pred_state.shape[1], -1, 3])
        true_points = tf.reshape(true_state, [true_state.shape[0], true_state.shape[1], -1, 3])
        all_distances = tf.linalg.norm(pred_points - true_points, axis=3)
        batch_position_error = tf.reduce_mean(all_distances)
        final_position_error = tf.reduce_mean(all_distances[:, -1])

        metrics['{} full error'.format(state_key)] = batch_position_error
        metrics['{} final error'.format(state_key)] = final_position_error
        for point_idx in range(all_distances.shape[2]):
            final_position_error_pt = tf.reduce_mean(all_distances[:, -1, point_idx])
            metrics['{} final error point #{}'.format(state_key, point_idx)] = final_position_error_pt
    return metrics
