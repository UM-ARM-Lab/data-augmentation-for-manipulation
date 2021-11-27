from typing import Dict

import tensorflow as tf
from tensorflow.keras import layers

from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from moonshine.metrics import LossMetric
from moonshine.my_keras_model import MyKerasModel


def compute_transformation_invariance_error(inputs, scenario):
    state_after_aug = inputs['state_after_aug']
    state_after_aug_expected = inputs['state_after_aug_expected']
    true_error = scenario.classifier_distance(state_after_aug_expected, state_after_aug)
    return true_error


class InvarianceModel(MyKerasModel):

    def __init__(self, hparams: Dict, batch_size: int, scenario: ScenarioWithVisualization):
        super().__init__(hparams, batch_size)
        self.scenario = scenario

        fc_layers = []
        for layer_size in self.hparams['fc_layer_sizes']:
            fc_layers.append(layers.Dense(layer_size,
                                          kernel_regularizer=tf.keras.regularizers.l2(self.hparams['reg']),
                                          bias_regularizer=tf.keras.regularizers.l2(self.hparams['reg']),
                                          activation='relu'))
        fc_layers.append(layers.Dense(1,
                                      kernel_regularizer=tf.keras.regularizers.l2(self.hparams['reg']),
                                      bias_regularizer=tf.keras.regularizers.l2(self.hparams['reg']),
                                      activation='relu'))
        self.sequential = tf.keras.Sequential(fc_layers)

        self.inputs_keys = ['transform']

    def compute_loss(self, inputs, outputs):
        true_error = inputs['error']
        true_error = tf.expand_dims(true_error, axis=-1)
        loss = tf.losses.MSE(true_error, outputs['predicted_error'])
        return {
            'loss': loss,
        }

    def create_metrics(self):
        super().create_metrics()
        return {
            'loss': LossMetric(),
        }

    # @tf.function
    def call(self, inputs: Dict, training, **kwargs):
        predicted_error = self.sequential(inputs['transform'])
        return {
            'predicted_error': predicted_error,
        }
