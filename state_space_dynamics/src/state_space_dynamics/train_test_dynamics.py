#!/usr/bin/env python
import pathlib
from typing import List, Optional, Callable

import hjson
import numpy as np
import tensorflow as tf
from colorama import Fore

import rospy
import state_space_dynamics
from link_bot_data.dataset_utils import batch_tf_dataset, deserialize_scene_msg
from link_bot_data.load_dataset import get_dynamics_dataset_loader
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine import filepath_tools, common_train_hparams
from moonshine.indexing import index_time_batched
from moonshine.metrics import LossMetric
from moonshine.model_runner import ModelRunner
from moonshine.moonshine_utils import remove_batch
from moonshine.numpify import numpify
from state_space_dynamics import dynamics_utils


def train_main(dataset_dirs: List[pathlib.Path],
               model_hparams: pathlib.Path,
               log: str,
               batch_size: int,
               epochs: int,
               seed: int,
               checkpoint: Optional[pathlib.Path] = None,
               ensemble_idx: Optional[int] = None,
               take: Optional[int] = None,
               trials_directory=pathlib.Path,
               ):
    print(Fore.CYAN + f"Using seed {seed}")

    model_hparams = hjson.load(model_hparams.open('r'))
    model_class = state_space_dynamics.get_model(model_hparams['model_class'])

    train_dataset = get_dynamics_dataset_loader(dataset_dirs)
    val_dataset = get_dynamics_dataset_loader(dataset_dirs)

    model_hparams.update(setup_hparams(batch_size, dataset_dirs, seed, train_dataset))
    model = model_class(hparams=model_hparams, batch_size=batch_size, scenario=train_dataset.scenario)

    trial_path = setup_training_paths(checkpoint, log, model_hparams, trials_directory, ensemble_idx)

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
    if checkpoint:
        trial_path = checkpoint.parent.absolute()
    group_name = log if trial_path is None else None
    if ensemble_idx is not None:
        group_name = f"{group_name}_{ensemble_idx}"
    trial_path, _ = filepath_tools.create_or_load_trial(group_name=group_name, trial_path=trial_path,
                                                        params=model_hparams, trials_directory=trials_directory)
    return trial_path


def setup_hparams(batch_size, dataset_dirs, seed, train_dataset):
    hparams = common_train_hparams.setup_hparams(batch_size, dataset_dirs, seed, train_dataset)
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
                                 ):
    test_dataset = get_dynamics_dataset_loader(dataset_dirs)

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
              ):
    dataset_loader = get_dynamics_dataset_loader(dataset_dirs)

    trials_directory = pathlib.Path('trials').absolute()
    trial_path = checkpoint.parent.absolute()
    _, params = filepath_tools.create_or_load_trial(trial_path=trial_path, trials_directory=trials_directory)
    model = state_space_dynamics.get_model(params['model_class'])
    scenario = dataset_loader.get_scenario()
    net = model(hparams=params, batch_size=batch_size, scenario=scenario)

    runner = ModelRunner(model=net,
                         training=False,
                         checkpoint=checkpoint,
                         batch_metadata=dataset_loader.batch_metadata,
                         trial_path=trial_path,
                         params=params)

    test_dataset = dataset_loader.get_datasets(mode=mode)
    test_dataset = batch_tf_dataset(test_dataset, batch_size, drop_remainder=True)
    val_metrics = {
        'loss': LossMetric(),
    }
    def _remove_scene_msg(e):
        e.pop("scene_msg")
        return e
    test_dataset = test_dataset.map(_remove_scene_msg)
    runner.val_epoch(test_dataset, val_metrics)
    for name, value in val_metrics.items():
        print(f"{name}: {value.result():.4f}")

    # more metrics that can't be expressed as just an average over metrics on each batch
    all_errors = None
    for batch in test_dataset:
        outputs = runner.model(batch, training=False)
        errors_for_batch = scenario.classifier_distance(outputs, batch)
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
                **kwargs,
                ):
    loader = get_dynamics_dataset_loader(dataset_dirs)

    dataset = loader.get_datasets(mode=mode).batch(1)

    model = dynamics_utils.load_generic_model(checkpoint)

    for i, e in enumerate(dataset):
        e.update(loader.batch_metadata)
        outputs = model.propagate_from_example(e, training=False)
        if isinstance(outputs, tuple):
            outputs, _ = outputs

        viz_func(e, outputs, loader, model)


def viz_example(example, outputs, loader, model):
    threshold = 0.1
    rospy.loginfo_once(f"Using {threshold=}")

    deserialize_scene_msg(example)
    s = loader.get_scenario()
    s.plot_environment_rviz(remove_batch(example))
    anim = RvizAnimationController(np.arange(loader.steps_per_traj))
    while not anim.done:
        t = anim.t()
        actual_t = loader.index_time_batched(example, t)
        s.plot_state_rviz(actual_t, label='viz_actual', color='red')
        s.plot_action_rviz(actual_t, actual_t, color='gray', label='viz')

        model_state_keys = model.state_keys + model.state_metadata_keys
        prediction_t = numpify(remove_batch(index_time_batched(outputs, model_state_keys, t, False)))
        s.plot_state_rviz(prediction_t, label='viz_predicted', color='blue')

        error_t = s.classifier_distance(actual_t, prediction_t)

        s.plot_error_rviz(error_t)
        label_t = error_t < threshold
        s.plot_is_close(label_t)

        anim.step()
