import tensorflow as tf
import tensorflow_probability as tfp


class ClassifierAugmentation:
    def __init__(self, hparams):
        self.aug_hparams = hparams.get('augmentation', None)
        self.aug_gen = tf.random.Generator.from_seed(0)
        self.aug_seed_stream = tfp.util.SeedStream(1, salt="nn_classifier_aug")
        self.aug_opt = tf.keras.optimizers.SGD(0.1)
        self.aug_opt_grad_norm_threshold = 0.008  # stopping criteria for the eng aug optimization
        self.barrier_upper_cutoff = tf.square(0.04)  # stops repelling points from pushing after this distance
        self.barrier_scale = 1.1  # scales the gradients for the repelling points
        self.env_aug_grad_clip = 5.0  # max dist step the env aug update can take
