import numpy as np
import tensorflow as tf


class RasterPoints(tf.keras.layers.Layer):

    def __init__(self, sdf_shape, **kwargs):
        super(RasterPoints, self).__init__(**kwargs)
        self.sdf_shape = sdf_shape
        self.n = None
        self.n_points = None

    def build(self, input_shapes):
        super(RasterPoints, self).build(input_shapes)
        self.n = int(input_shapes[0][1])
        self.n_points = int(self.n // 2)

    def raster_points(self, rope_configurations, resolution, origin):
        rope_configurations = np.atleast_2d(rope_configurations)
        batch_size = rope_configurations.shape[0]
        rope_images = np.zeros([batch_size, self.sdf_shape[0], self.sdf_shape[1], self.n_points],
                               dtype=np.float32)
        batch_resolution = resolution
        batch_origin = origin
        row_y_indeces = tf.cast(rope_configurations[:, :, 1] / batch_resolution[:, 0:1] + batch_origin[:, 0:1], tf.int64)
        col_x_indeces = tf.cast(rope_configurations[:, :, 0] / batch_resolution[:, 1:2] + batch_origin[:, 1:2], tf.int64)
        batch_indeces = np.arange(batch_size).repeat(self.n_points)
        row_indeces = tf.flatten(row_y_indeces)
        col_indeces = tf.flatten(col_x_indeces)
        point_channel_indeces = np.tile(np.arange(self.n_points), batch_size)
        rope_images[batch_indeces, row_indeces, col_indeces, point_channel_indeces] = 1
        return rope_images

    def call(self, inputs, **kwargs):
        """
        :param x: [sequence_length, n_points * 2], [sequence_length, 2], [sequence_length, 2]
        :return: sdf_shape
        """
        x, resolution, origin = inputs
        points = tf.reshape(x, [-1, self.n_points, 2], name='points_reshape')
        rope_images = tf.py_function(self.raster_points, [points, resolution, origin], tf.float32, name='raster_points')
        input_shapes = [input.shape for input in inputs]
        rope_images.set_shape(self.compute_output_shape(input_shapes))
        return rope_images

    def get_config(self):
        config = {}
        config.update(super(RasterPoints, self).get_config())
        return config

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], self.sdf_shape[0], self.sdf_shape[1], self.n_points
