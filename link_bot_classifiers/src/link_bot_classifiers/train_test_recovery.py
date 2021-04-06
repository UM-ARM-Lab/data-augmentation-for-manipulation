#!/usr/bin/env python
import json
import pathlib
import time
from typing import Optional, List

import tensorflow as tf

from link_bot_classifiers.nn_recovery_policy import NNRecoveryModel
from link_bot_data.dataset_utils import batch_tf_dataset
from link_bot_data.recovery_dataset import RecoveryDatasetLoader
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.pycommon import paths_to_json
from moonshine import filepath_tools
from moonshine.model_runner import ModelRunner
from moonshine.moonshine_utils import restore_variables


def train_main(dataset_dirs: List[pathlib.Path],
               model_hparams: pathlib.Path,
               classifier_checkpoint: pathlib.Path,
               log: str,
               batch_size: int,
               epochs: int,
               seed: int,
               checkpoint: Optional[pathlib.Path] = None,
               **kwargs,
               ):
    ###############
    # Datasets
    ###############
    train_dataset = RecoveryDatasetLoader(dataset_dirs)
    val_dataset = RecoveryDatasetLoader(dataset_dirs)

    ###############
    # Model
    ###############
    model_hparams = json.load((model_hparams).open('r'))
    model_hparams['recovery_dataset_hparams'] = train_dataset.hparams
    model_hparams['batch_size'] = batch_size
    model_hparams['seed'] = seed
    model_hparams['datasets'] = paths_to_json(dataset_dirs)
    model_hparams['latest_training_time'] = int(time.time())
    scenario = get_scenario(model_hparams['scenario'])

    # Dataset preprocessing
    train_tf_dataset = train_dataset.get_datasets(mode='train')
    val_tf_dataset = val_dataset.get_datasets(mode='val')

    train_tf_dataset = batch_tf_dataset(train_tf_dataset, batch_size, drop_remainder=True)
    val_tf_dataset = batch_tf_dataset(val_tf_dataset, batch_size, drop_remainder=True)

    train_tf_dataset = train_tf_dataset.shuffle(buffer_size=512, seed=seed)

    train_tf_dataset = train_tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    val_tf_dataset = val_tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    model = NNRecoveryModel(hparams=model_hparams, scenario=scenario, batch_size=batch_size)

    ############
    # Initialize weights from classifier model by "restoring" from checkpoint
    ############
    if not checkpoint:
        # load in the weights for the conv layers of the classifier's encoder, skip the last few layers
        restore_variables(classifier_checkpoint, conv_layers=model.conv_layers)
    ############

    trial_path = None
    if checkpoint:
        trial_path = checkpoint.parent.absolute()
    trials_directory = pathlib.Path('recovery_trials').absolute()
    group_name = log if trial_path is None else None
    trial_path, _ = filepath_tools.create_or_load_trial(group_name=group_name,
                                                        params=model_hparams,
                                                        trial_path=trial_path,
                                                        trials_directory=trials_directory,
                                                        write_summary=False)
    runner = ModelRunner(model=model,
                         training=True,
                         params=model_hparams,
                         trial_path=trial_path,
                         val_every_n_batches=1,
                         mid_epoch_val_batches=100,
                         validate_first=True,
                         checkpoint=checkpoint,
                         batch_metadata=train_dataset.batch_metadata)

    # Train
    runner.train(train_tf_dataset, val_tf_dataset, num_epochs=epochs)

    return trial_path


def eval_main(dataset_dirs: List[pathlib.Path],
              checkpoint: pathlib.Path,
              mode: str,
              batch_size: int,
              **kwargs,
              ):
    ###############
    # Model
    ###############
    trial_path = checkpoint.parent.absolute()
    trials_directory = pathlib.Path('recovery_trials').absolute()
    _, params = filepath_tools.create_or_load_trial(trial_path=trial_path,
                                                    trials_directory=trials_directory)
    scenario = get_scenario(params['scenario'])
    net = NNRecoveryModel(hparams=params, scenario=scenario, batch_size=1)

    ###############
    # Dataset
    ###############
    test_dataset = RecoveryDatasetLoader(dataset_dirs)
    test_tf_dataset = test_dataset.get_datasets(mode=mode)

    ###############
    # Evaluate
    ###############
    test_tf_dataset = batch_tf_dataset(test_tf_dataset, batch_size, drop_remainder=True)

    runner = ModelRunner(model=net,
                         training=False,
                         params=params,
                         checkpoint=checkpoint,
                         trial_path=trial_path,
                         batch_metadata=test_dataset.batch_metadata)
    validation_metrics = runner.val_epoch(test_tf_dataset)
    for name, value in validation_metrics.items():
        print(f"{name}: {value:.3f}")
