from typing import Dict

import tensorflow as tf
from tensorflow.keras import layers

from moonshine.metrics import LossMetric
from moonshine.my_keras_model import MyKerasModel


class InvarianceModel(MyKerasModel):

    def __init__(self, hparams: Dict, batch_size: int):
        super().__init__(hparams, batch_size)
        self.scenario = get_scenario()

        fc_layers = []
        for layer_size in self.hparams['fc_layer_sizes']:
            fc_layers.append(layers.Dense(layer_size,
                                          kernel_regularizer=self.hparams['reg'],
                                          bias_regularizer=self.hparams['reg']))
            fc_layers.append(layers.ReLU())
        self.sequential = tf.keras.Sequential(fc_layers)

    def compute_loss(self, dataset_element, outputs):
        loss = tf.losses.MSE(outputs['true_error'], outputs['predicted_error'])
        return {
            'loss': loss,
        }

    def create_metrics(self):
        super().create_metrics()
        return {
            'loss': LossMetric(),
        }

    @tf.function
    def call(self, inputs: Dict, training, **kwargs):
        input_vec = tf.concat()
        predicted_error = self.sequential(input_vec)

        state_after_aug = inputs['state_after_aug']
        state_after_aug_expected = inputs['state_after_aug_expected']
        true_error = self.scenario.classifier_distance(state_after_aug_expected, state_after_aug)

        return {
            'true_error':      true_error,
            'predicted_error': predicted_error,
        }
