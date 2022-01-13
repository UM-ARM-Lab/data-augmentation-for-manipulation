from copy import deepcopy
from typing import Dict

import tensorflow as tf
from colorama import Fore
from tensorflow.keras.metrics import Metric

from moonshine.metrics import LossMetric


class MyKerasModel(tf.keras.Model):

    def get_config(self):
        super_config = super(self).get_config()
        super_config.update({
            'hparams':    self.hparams,
            'batch_size': self.batch_size,
        })
        return super_config

    def __init__(self, hparams, batch_size, verbose: int = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.verbose = verbose
        self.hparams = deepcopy(hparams)
        learning_rate = hparams.get('learning_rate', 1e-3)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def call(self, dataset_element, training=False, **kwargs):
        """
        The forward pass. The output here gets passed directly to apply_gradients and compute_loss
        :param inputs: an element of the dataset
        :param training: true if training, used for things like batchnorm and dropout.
        :param kwargs:
        :return: anything needed to compute loss & update gradients, potentially a tuple or dictionary.
        """
        raise NotImplementedError()

    def compute_loss(self, dataset_element, outputs):
        """
        Computes loss and returns a dictionary
        :param dataset_element: an element of the dataset
        :param outputs: the output of the forward pass, call(), and is often a dictionary
        :return: a dictionary of losses where keys control the names in tensorboard
        """
        raise NotImplementedError()

    def compute_metrics(self, metrics: Dict[str, Metric], dataset_element, outputs):
        return {}

    # No tf.function is needed here, since train_step is decorated
    # adding tf.function here kills the gradients for some unknown reason, something to due with "losses" being passed in
    # potentially it gets copied?
    def apply_gradients(self, tape, train_element, train_outputs, losses, metrics: Dict[str, Metric]):
        """
        Applies gradients to the optimizers and returns metrics for losses and gradients
        :param tape: gradient tape
        :param train_element: an element of the dataset
        :param train_outputs: the output of the forward pass, call(), and is often a dictionary
        :param losses: a dictionary of losses at least containing one key 'loss' for the total training loss
        :return: a dictionary of metrics where keys control the names in tensorboard
        """
        # the 'loss' key is assumed to be the total loss used for training
        train_batch_loss = losses['loss']
        variables = self.trainable_variables
        gradients = tape.gradient(train_batch_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return {}

    def preprocess_no_gradient(self, element, training: bool):
        return element

    def train_step(self, train_element, metrics: Dict[str, Metric]):
        train_element = self.preprocess_no_gradient(train_element, training=True)
        return self._train_step(train_element, metrics)

    # @tf.function
    def _train_step(self, train_element, metrics: Dict[str, Metric]):
        with tf.GradientTape(persistent=True) as tape:
            train_outputs = self.call(train_element, training=True)
            train_losses = self.compute_loss(train_element, train_outputs)

        self.apply_gradients(tape, train_element, train_outputs, train_losses, metrics)

        self.update_metrics(metrics, train_element, train_losses, train_outputs)

        return train_outputs

    def val_step(self, val_element, metrics: Dict[str, Metric]):
        val_element = self.preprocess_no_gradient(val_element, training=False)
        return self._val_step(val_element, metrics)

    def _val_step(self, val_element, metrics: Dict[str, Metric]):
        val_outputs = self.call(val_element, training=False)
        val_losses = self.compute_loss(val_element, val_outputs)

        self.update_metrics(metrics, val_element, val_losses, val_outputs)

        return val_outputs

    def update_metrics(self, metrics, inputs, losses, outputs):
        self.compute_metrics(metrics, inputs, outputs)
        for loss_name_k, batch_loss_k in losses.items():
            metrics[loss_name_k].update_state(batch_loss_k)

    def create_metrics(self):
        if self.verbose > -1:
            print(Fore.YELLOW + "Creating Metrics")
        return {
            'loss': LossMetric(),
        }

    def on_end_epoch(self):
        pass

    def on_mid_epoch_validation(self):
        pass
