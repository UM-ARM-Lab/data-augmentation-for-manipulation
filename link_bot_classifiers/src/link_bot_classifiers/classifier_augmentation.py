import pathlib
from typing import Optional

import tensorflow as tf
import tensorflow_probability as tfp

from learn_invariance.invariance_model_wrapper import InvarianceModelWrapper
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization


class ClassifierAugmentation:
    def __init__(self, hparams, batch_size: int, scenario: ScenarioWithVisualization):
        self.hparams = hparams.get('augmentation', None)
        self.batch_size = batch_size
        self.scenario = scenario

        self.gen = tf.random.Generator.from_seed(0)
        self.seed = tfp.util.SeedStream(1, salt="nn_classifier_aug")
        self.opt = tf.keras.optimizers.SGD(0.1)
        self.grad_norm_threshold = 0.01  # stopping criteria for the eng aug optimization
        self.barrier_upper_lim = tf.square(0.06)  # stops repelling points from pushing after this distance
        self.barrier_scale = 0.05  # scales the gradients for the repelling points
        self.grad_clip = 5.0  # max dist step the env aug update can take

        if self.hparams is not None:
            invariance_model_path = pathlib.Path(self.hparams['invariance_model'])
            self.invariance_model_wrapper = InvarianceModelWrapper(invariance_model_path, self.batch_size,
                                                                   self.scenario)

    def do_augmentation(self):
        return self.hparams is not None
