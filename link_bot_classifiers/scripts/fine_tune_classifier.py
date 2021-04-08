#!/usr/bin/env python
import argparse
import logging
import pathlib
from typing import List

import colorama
import numpy as np
import tensorflow as tf

import link_bot_classifiers
from arc_utilities import ros_init
from link_bot_classifiers.train_test_classifier import setup_datasets
from link_bot_data.classifier_dataset import ClassifierDatasetLoader
from link_bot_pycommon.pycommon import paths_to_json
from moonshine.filepath_tools import load_trial, create_trial
from moonshine.model_runner import ModelRunner


@ros_init.with_ros("fine_tune_classifier")
def main():
    colorama.init(autoreset=True)
    tf.get_logger().setLevel(logging.ERROR)

    np.set_printoptions(linewidth=250, precision=4, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    parser.add_argument('checkpoint', type=pathlib.Path)
    parser.add_argument('log')
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--log-scalars-every', type=int, help='loss/accuracy every this many batches', default=100)
    parser.add_argument('--validation-every', type=int, help='report validation every this many epochs', default=1)
    parser.add_argument('--threshold', type=float, default=None)

    np.random.seed(1)
    tf.random.set_seed(1)

    args = parser.parse_args()

    fine_tune_classifier(dataset_dirs=args.dataset_dirs,
                         checkpoint=args.checkpoint,
                         log=args.log,
                         batch_size=args.batch_size,
                         epochs=args.epochs)


def fine_tune_classifier(dataset_dirs: List[pathlib.Path],
                         checkpoint: pathlib.Path,
                         log: str,
                         batch_size: int,
                         epochs: int,
                         trials_directory: pathlib.Path = pathlib.Path("./trials")):
    _, model_hparams = load_trial(trial_path=checkpoint.parent.absolute())
    model_hparams['datasets'].extend(paths_to_json(dataset_dirs))

    trial_path, _ = create_trial(log, model_hparams, trials_directory=trials_directory)

    model_class = link_bot_classifiers.get_model(model_hparams['model_class'])

    train_dataset = ClassifierDatasetLoader(dataset_dirs, use_gt_rope=True, load_true_states=True)
    val_dataset = ClassifierDatasetLoader(dataset_dirs, use_gt_rope=True, load_true_states=True)

    # decrease the learning rate, this is often done in fine-tuning
    model_hparams['learning_rate'] = 1e-4  # normally 1e-3
    model = model_class(hparams=model_hparams, batch_size=batch_size, scenario=train_dataset.scenario)
    runner = ModelRunner(model=model,
                         training=True,
                         params=model_hparams,
                         checkpoint=checkpoint,
                         batch_metadata=train_dataset.batch_metadata,
                         validate_first=True,
                         val_every_n_batches=500,
                         mid_epoch_val_batches=100,
                         trial_path=trial_path)

    train_tf_dataset, val_tf_dataset = setup_datasets(model_hparams, batch_size, train_dataset, val_dataset)

    # Modify the model for feature transfer & fine-tuning
    for c in model.conv_layers:
        c.trainable = False
    for d in model.dense_layers:
        d.trainable = False
    model.lstm.trainable = False

    runner.reset_best_ket_metric_value()
    runner.train(train_tf_dataset, val_tf_dataset, num_epochs=epochs)

    return trial_path


if __name__ == '__main__':
    main()
