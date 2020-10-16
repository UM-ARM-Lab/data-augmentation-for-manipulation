from typing import Callable

import tensorflow as tf


# CuRL and other papers suggest that random crop is the most important data augmentation for learning state/dynamics
# https://arxiv.org/pdf/2004.13649.pdf
# https://www.tensorflow.org/tutorials/images/data_augmentation
def augment(image_sequence, image_h: int, image_w: int, generator: tf.random.Generator):
    def _random_crop(images):
        # Add 6 pixels of padding
        images_padded = tf.pad(images, [[0, 0], [6, 6], [6, 6], [0, 0]])

        # Random crop back to the original size. I believe this crops every image in the batch the same way, which should be fine
        # since the batches are assembled randomly, and is good because then the cropping is consistent over time,
        # since time is currently shoved into batch dimension for these operations
        random_crop_shape = images_padded.shape[:-3].as_list() + [image_h, image_w] + [images_padded.shape[-1]]
        out_images = tf.image.random_crop(images_padded, size=random_crop_shape)
        return out_images

    return apply_to_image_sequence(image_sequence, _random_crop)


def resize_image_sequence(image_sequence, image_h: int, image_w: int):
    def _resize(images):
        return tf.image.resize(images, [image_h, image_w], preserve_aspect_ratio=True)

    return apply_to_image_sequence(image_sequence, _resize)


def apply_to_image_sequence(image_sequence, func: Callable):
    """
    takes a callable function of one argument, an image tensor of shape [B, H, W, C], and applies to to the imput tensor image_sequence
     which is assumed to be of shape [B1, B2, ..., BN, H, W, C]
     """
    one_batch_image, original_batch_dims = flatten_batch_and_sequence(image_sequence)

    out_image = func(one_batch_image)

    out_image_sequence = unflatten_batch_and_sequence(out_image, original_batch_dims)
    return out_image_sequence


def flatten_batch_and_sequence(x):
    """ assume x is [batch, time, a, b, ... ] """
    original_shape = x.shape
    original_batch_dims = original_shape[:-3].as_list()
    image_shape = original_shape[-3:].as_list()
    x_flat = tf.reshape(x, [-1] + image_shape)
    # monkey patch
    return x_flat, original_batch_dims


def unflatten_batch_and_sequence(x_flat, original_batch_dims):
    out_dims = x_flat.shape[1:].as_list()
    x = tf.reshape(x_flat, original_batch_dims + out_dims)
    return x
