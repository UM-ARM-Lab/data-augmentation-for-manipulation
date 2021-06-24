#!/usr/bin/env python
import argparse
import logging
import pathlib

import numpy as np
import tensorflow as tf

from arc_utilities import ros_init
from link_bot_classifiers.fine_tune_classifier import fine_tune_classifier
from link_bot_pycommon.args import int_tuple_arg
from moonshine.filepath_tools import load_hjson


@ros_init.with_ros("fine_tune_classifier")
def main():
    tf.get_logger().setLevel(logging.ERROR)

    np.set_printoptions(linewidth=250, precision=4, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    parser.add_argument('checkpoint', type=pathlib.Path)
    parser.add_argument('log')
    parser.add_argument('--params', '-p', type=pathlib.Path, help='an hjson file to override the model hparams')
    parser.add_argument('--pretransfer-config-dir', type=pathlib.Path, help='dir of pkl files with state/env')
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--profile', type=int_tuple_arg, default=None)
    parser.add_argument('--take', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--log-scalars-every', type=int, help='loss/accuracy every this many batches', default=100)
    parser.add_argument('--validation-every', type=int, help='report validation every this many epochs', default=1)
    parser.add_argument('--no-validate-first', action='store_true')
    parser.add_argument('--threshold', type=float, default=None)

    args = parser.parse_args()

    if args.params is not None:
        model_hparams_update = load_hjson(args.params)
    else:
        model_hparams_update = None

    fine_tune_classifier(dataset_dirs=args.dataset_dirs,
                         checkpoint=args.checkpoint,
                         log=args.log,
                         batch_size=args.batch_size,
                         early_stopping=False,
                         epochs=args.epochs,
                         validate_first=(not args.no_validate_first),
                         take=args.take,
                         seed=args.seed,
                         model_hparams_update=model_hparams_update,
                         val_every_n_batches=500,
                         mid_epoch_val_batches=100,
                         fine_tune_conv=False,
                         fine_tune_lstm=False,
                         fine_tune_dense=False,
                         fine_tune_output=True,
                         augmentation_config_dir=args.pretransfer_config_dir,
                         profile=args.profile,
                         )


if __name__ == '__main__':
    main()
