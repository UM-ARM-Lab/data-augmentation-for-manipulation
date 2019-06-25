#!/usr/bin/env python
from __future__ import print_function

import argparse
import sys

import numpy as np
import tensorflow as tf
from keras.layers import Input, Concatenate, Dense
from keras.models import Model

from link_bot_models.base_model import BaseModel
from link_bot_models.components.simple_cnn_model import SimpleCNNModel
from link_bot_models.label_types import LabelType
from link_bot_models.multi_environment_datasets import MultiEnvironmentDataset
from link_bot_pycommon import experiments_util

simple_cnn_label_types = [LabelType.SDF]


class SimpleCNNModelRunner(BaseModel):

    def __init__(self, args_dict, sdf_shape, N):
        super(SimpleCNNModelRunner, self).__init__(args_dict, N)
        self.sdf_shape = sdf_shape

        sdf_input = Input(shape=(sdf_shape[0], sdf_shape[1], 1), dtype='float32', name='sdf')
        rope_input = Input(shape=(N,), dtype='float32', name='rope_configuration')

        self.conv_filters = [
            (64, (7, 7)),
            (32, (5, 5)),
            (16, (3, 3)),
            (16, (3, 3)),
        ]

        self.fc_layer_sizes = [
            256,
            32,
        ]

        cnn_output = SimpleCNNModel(self.conv_filters, [], output_dim=128)(sdf_input)
        concat = Concatenate()([cnn_output, rope_input])

        fc_h = concat
        for fc_layer_size in self.fc_layer_sizes:
            fc_h = Dense(fc_layer_size, activation='relu')(fc_h)
        prediction = Dense(1, activation='sigmoid', name='combined_output')(fc_h)

        # This creates a model that includes
        self.model_inputs = [sdf_input, rope_input]
        self.keras_model = Model(inputs=self.model_inputs, outputs=prediction)
        self.keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def metadata(self, label_types):
        extra_metadata = {
            'conv_filters': self.conv_filters,
            'fc_layer_sizes': self.fc_layer_sizes,
            'sdf_shape': self.sdf_shape,
        }
        extra_metadata.update(super(SimpleCNNModelRunner, self).metadata(label_types))
        return extra_metadata

    def violated(self, observations, sdf_data):
        m = observations.shape[0]
        rope_configuration = observations
        sdf = np.tile(np.expand_dims(sdf_data.sdf, axis=2), [m, 1, 1, 1])
        sdf_gradient = np.tile(sdf_data.gradient, [m, 1, 1, 1])
        sdf_origin = np.tile(sdf_data.origin, [m, 1])
        sdf_resolution = np.tile(sdf_data.resolution, [m, 1])
        sdf_extent = np.tile(sdf_data.extent, [m, 1])
        inputs_dict = {
            'rope_configuration': rope_configuration,
            'sdf': sdf,
            'sdf_gradient': sdf_gradient,
            'sdf_origin': sdf_origin,
            'sdf_resolution': sdf_resolution,
            'sdf_extent': sdf_extent
        }

        predicted_violated = (self.keras_model.predict(inputs_dict) > 0.5).astype(np.bool)
        return predicted_violated


def train(args):
    log_path = experiments_util.experiment_name(args.log)

    train_dataset = MultiEnvironmentDataset.load_dataset(args.train_dataset)
    validation_dataset = MultiEnvironmentDataset.load_dataset(args.validation_dataset)
    sdf_shape = train_dataset.sdf_shape

    if args.checkpoint:
        model = SimpleCNNModelRunner.load(vars(args), sdf_shape, args.N)
    else:
        model = SimpleCNNModelRunner(vars(args), sdf_shape, args.N)

    model.train(train_dataset, validation_dataset, simple_cnn_label_types, args.epochs, log_path)


def evaluate(args):
    dataset = MultiEnvironmentDataset.load_dataset(args.dataset)
    sdf_shape = dataset.sdf_shape

    model = SimpleCNNModelRunner.load(vars(args), sdf_shape, args.N)

    return model.evaluate(dataset, simple_cnn_label_types)


def main():
    np.set_printoptions(precision=6, suppress=True)
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("-N", help="dimensions in input state", type=int, default=6)
    parser.add_argument("--debug", help="enable TF Debugger", action='store_true')
    parser.add_argument("--seed", type=int, default=0)

    subparsers = parser.add_subparsers()
    train_subparser = subparsers.add_parser("train")
    train_subparser.add_argument("train_dataset", help="dataset (json file)")
    train_subparser.add_argument("validation_dataset", help="dataset (json file)")
    train_subparser.add_argument("--batch-size", "-b", type=int, default=100)
    train_subparser.add_argument("--log", "-l", nargs='?', help="save/log the graph and summaries", const="")
    train_subparser.add_argument("--epochs", "-e", type=int, help="number of epochs to train for", default=250)
    train_subparser.add_argument("--checkpoint", "-c", help="restart from this *.ckpt name")
    train_subparser.add_argument("--plot", action="store_true")
    train_subparser.add_argument("--val-acc-threshold", type=float, default=None)
    train_subparser.add_argument("--skip-validation", action='store_true')
    train_subparser.add_argument("--early-stopping", action='store_true')
    train_subparser.set_defaults(func=train)

    eval_subparser = subparsers.add_parser("eval")
    eval_subparser.add_argument("dataset", help="dataset (json file)")
    eval_subparser.add_argument("checkpoint", help="eval the *.ckpt name")
    eval_subparser.add_argument("--batch-size", "-b", type=int, default=100)
    eval_subparser.set_defaults(func=evaluate)

    args = parser.parse_args()
    commandline = ' '.join(sys.argv)
    args.commandline = commandline

    np.random.seed(args.seed)
    tf.random.set_random_seed(args.seed)

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
