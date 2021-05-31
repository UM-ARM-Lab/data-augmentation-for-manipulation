#!/usr/bin/env python
import pathlib
from typing import List, Optional, Callable

import hjson
import numpy as np
import tensorflow as tf
from colorama import Fore

import rospy
import state_space_dynamics
from link_bot_data.dataset_utils import batch_tf_dataset
from link_bot_data.dynamics_dataset import DynamicsDatasetLoader
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine import filepath_tools, common_train_hparams
from moonshine.model_runner import ModelRunner
from moonshine.moonshine_utils import remove_batch
from state_space_dynamics import dynamics_utils


def train_main(dataset_dirs: List[pathlib.Path],
               model_hparams: pathlib.Path,
               log: str,
               batch_size: int,
               epochs: int,
               seed: int,
               use_gt_rope: bool,
               checkpoint: Optional[pathlib.Path] = None,
               ensemble_idx: Optional[int] = None,
               take: Optional[int] = None,
               trials_directory=pathlib.Path,
               ):
    print(Fore.CYAN + f"Using seed {seed}")

    model_hparams = hjson.load(model_hparams.open('r'))
    model_class = state_space_dynamics.get_model(model_hparams['model_class'])

    train_dataset = DynamicsDatasetLoader(dataset_dirs, use_gt_rope=use_gt_rope)
    val_dataset = DynamicsDatasetLoader(dataset_dirs, use_gt_rope=use_gt_rope)

    model_hparams.update(setup_hparams(batch_size, dataset_dirs, seed, train_dataset, use_gt_rope))
    model = model_class(hparams=model_hparams, batch_size=batch_size, scenario=train_dataset.scenario)

    checkpoint_name, trial_path = setup_training_paths(checkpoint, log, model_hparams, trials_directory, ensemble_idx)

    runner = ModelRunner(model=model,
                         training=True,
                         params=model_hparams,
                         checkpoint=checkpoint,
                         batch_metadata=train_dataset.batch_metadata,
                         trial_path=trial_path)

    train_tf_dataset, val_tf_dataset = setup_datasets(model_hparams, batch_size, train_dataset, val_dataset, take)

    runner.train(train_tf_dataset, val_tf_dataset, num_epochs=epochs)

    return trial_path


def setup_training_paths(checkpoint, log, model_hparams, trials_directory, ensemble_idx=None):
    trial_path = None
    checkpoint_name = None
    if checkpoint:
        trial_path = checkpoint.parent.absolute()
        checkpoint_name = checkpoint.name
    group_name = log if trial_path is None else None
    if ensemble_idx is not None:
        group_name = f"{group_name}_{ensemble_idx}"
    trial_path, _ = filepath_tools.create_or_load_trial(group_name=group_name,
                                                        params=model_hparams,
                                                        trial_path=trial_path,
                                                        trials_directory=trials_directory,
                                                        write_summary=False)
    return checkpoint_name, trial_path


def setup_hparams(batch_size, dataset_dirs, seed, train_dataset, use_gt_rope):
    hparams = common_train_hparams.setup_hparams(batch_size, dataset_dirs, seed, train_dataset, use_gt_rope)
    hparams.update({
        'dynamics_dataset_hparams': train_dataset.params,
    })
    return hparams


def setup_datasets(model_hparams, batch_size, train_dataset, val_dataset, take: Optional[int] = None):
    # Dataset preprocessing
    train_tf_dataset = train_dataset.get_datasets(mode='train', take=take)
    val_tf_dataset = val_dataset.get_datasets(mode='val')

    # mix up examples before batching
    train_tf_dataset = train_tf_dataset.shuffle(model_hparams['shuffle_buffer_size'])

    train_tf_dataset = batch_tf_dataset(train_tf_dataset, batch_size, drop_remainder=True)
    val_tf_dataset = batch_tf_dataset(val_tf_dataset, batch_size, drop_remainder=True)

    train_tf_dataset = train_tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    val_tf_dataset = val_tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return train_tf_dataset, val_tf_dataset


def compute_classifier_threshold(dataset_dirs: List[pathlib.Path],
                                 checkpoint: pathlib.Path,
                                 mode: str,
                                 batch_size: int,
                                 use_gt_rope: bool,
                                 ):
    test_dataset = DynamicsDatasetLoader(dataset_dirs, use_gt_rope=use_gt_rope)

    trials_directory = pathlib.Path('trials').absolute()
    trial_path = checkpoint.parent.absolute()
    _, params = filepath_tools.create_or_load_trial(trial_path=trial_path, trials_directory=trials_directory)
    model = state_space_dynamics.get_model(params['model_class'])
    net = model(hparams=params, batch_size=batch_size, scenario=test_dataset.scenario)

    runner = ModelRunner(model=net,
                         training=False,
                         checkpoint=checkpoint,
                         batch_metadata=test_dataset.batch_metadata,
                         trial_path=trial_path,
                         params=params)

    test_tf_dataset = test_dataset.get_datasets(mode=mode)
    test_tf_dataset = batch_tf_dataset(test_tf_dataset, batch_size, drop_remainder=True)

    all_errors = None
    for batch in test_tf_dataset:
        outputs = runner.model(batch, training=False)
        errors_for_batch = test_dataset.scenario.classifier_distance(batch, outputs)
        if all_errors is not None:
            all_errors = tf.concat([all_errors, errors_for_batch], axis=0)
        else:
            all_errors = errors_for_batch

    classifier_threshold = np.percentile(all_errors.numpy(), 90)
    rospy.loginfo(f"90th percentile {classifier_threshold}")
    return classifier_threshold


def eval_main(dataset_dirs: List[pathlib.Path],
              checkpoint: pathlib.Path,
              mode: str,
              batch_size: int,
              use_gt_rope: bool,
              ):
    test_dataset = DynamicsDatasetLoader(dataset_dirs, use_gt_rope=use_gt_rope)

    trials_directory = pathlib.Path('trials').absolute()
    trial_path = checkpoint.parent.absolute()
    _, params = filepath_tools.create_or_load_trial(trial_path=trial_path, trials_directory=trials_directory)
    model = state_space_dynamics.get_model(params['model_class'])
    net = model(hparams=params, batch_size=batch_size, scenario=test_dataset.scenario)

    runner = ModelRunner(model=net,
                         training=False,
                         checkpoint=checkpoint,
                         batch_metadata=test_dataset.batch_metadata,
                         trial_path=trial_path,
                         params=params)

    test_tf_dataset = test_dataset.get_datasets(mode=mode)
    test_tf_dataset = batch_tf_dataset(test_tf_dataset, batch_size, drop_remainder=True)
    validation_metrics = runner.val_epoch(test_tf_dataset)
    for name, value in validation_metrics.items():
        print(f"{name}: {value}")

    # more metrics that can't be expressed as just an average over metrics on each batch
    all_errors = None
    for batch in test_tf_dataset:
        outputs = runner.model(batch, training=False)
        errors_for_batch = test_dataset.scenario.classifier_distance(outputs, batch)
        if all_errors is not None:
            all_errors = tf.concat([all_errors, errors_for_batch], axis=0)
        else:
            all_errors = errors_for_batch
    print(f"90th percentile {np.percentile(all_errors.numpy(), 90)}")
    print(f"95th percentile {np.percentile(all_errors.numpy(), 95)}")
    print(f"99th percentile {np.percentile(all_errors.numpy(), 99)}")
    print(f"max {np.max(all_errors.numpy())}")


def viz_main(dataset_dirs: List[pathlib.Path],
             checkpoint: pathlib.Path,
             mode: str,
             **kwargs):
    viz_dataset(dataset_dirs=dataset_dirs,
                checkpoint=checkpoint,
                mode=mode,
                viz_func=viz_example,
                **kwargs)


def viz_dataset(dataset_dirs: List[pathlib.Path],
                checkpoint: pathlib.Path,
                mode: str,
                viz_func: Callable,
                use_gt_rope: bool,
                **kwargs,
                ):
    test_dataset = DynamicsDatasetLoader(dataset_dirs, use_gt_rope=use_gt_rope)

    test_tf_dataset = test_dataset.get_datasets(mode=mode)
    test_tf_dataset = batch_tf_dataset(test_tf_dataset, 1, drop_remainder=True)

    model, _ = dynamics_utils.load_generic_model([checkpoint])

    for i, batch in enumerate(test_tf_dataset):
        batch.update(test_dataset.batch_metadata)
        outputs, _ = model.from_example(batch, training=False)

        viz_func(batch, outputs, test_dataset, model)


def viz_example(batch, outputs, test_dataset: DynamicsDatasetLoader, model):
    test_dataset.scenario.plot_environment_rviz(remove_batch(batch))
    anim = RvizAnimationController(np.arange(test_dataset.steps_per_traj))
    while not anim.done:
        t = anim.t()
        actual_t = test_dataset.index_time_batched(batch, t)
        test_dataset.scenario.plot_state_rviz(actual_t, label='actual', color='red')
        test_dataset.scenario.plot_action_rviz(actual_t, actual_t, color='gray')

        prediction_t = test_dataset.index_time_batched(outputs, t)
        test_dataset.scenario.plot_state_rviz(prediction_t, label='predicted', color='blue')

        anim.step()
