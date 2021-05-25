import tensorflow as tf
import tensorflow_probability as tfp


class ClassifierAugmentation:
    def __init__(self, hparams):
        self.hparams = hparams.get('augmentation', None)
        self.gen = tf.random.Generator.from_seed(0)
        self.seed = tfp.util.SeedStream(1, salt="nn_classifier_aug")
        self.opt = tf.keras.optimizers.SGD(0.1)
        self.grad_norm_threshold = 0.01  # stopping criteria for the eng aug optimization
        self.barrier_upper_lim = tf.square(0.06)  # stops repelling points from pushing after this distance
        self.barrier_scale = 0.05  # scales the gradients for the repelling points
        self.grad_clip = 5.0  # max dist step the env aug update can take

    def do_augmentation(self):
        return self.hparams is not None
