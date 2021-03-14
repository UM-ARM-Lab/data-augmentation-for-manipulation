#!/usr/bin/env python
import pathlib
from time import time
from typing import List, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from colorama import Fore
from progressbar import progressbar

import link_bot_classifiers
import rospy
from link_bot_classifiers import classifier_utils
from link_bot_classifiers.classifier_utils import load_generic_model
from link_bot_classifiers.nn_classifier import NNClassifierWrapper
from link_bot_data import base_dataset
from link_bot_data.classifier_dataset import ClassifierDatasetLoader
from link_bot_data.dataset_utils import add_predicted, batch_tf_dataset
from link_bot_data.visualization import init_viz_env
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from link_bot_pycommon.serialization import my_hdump
from merrrt_visualization.rviz_animation_controller import RvizAnimation
from moonshine import filepath_tools
from moonshine.ensemble import Ensemble2
from moonshine.filepath_tools import load_hjson
from moonshine.indexing import index_dict_of_batched_tensors_tf
from moonshine.metrics import AccuracyCheckpointMetric
from moonshine.model_runner import ModelRunner
from moonshine.moonshine_utils import numpify
from state_space_dynamics import common_train_hparams
from state_space_dynamics.train_test import setup_training_paths
from std_msgs.msg import Float32


def setup_hparams(batch_size, dataset_dirs, seed, train_dataset, use_gt_rope):
    hparams = common_train_hparams.setup_hparams(batch_size, dataset_dirs, seed, train_dataset, use_gt_rope)
    hparams.update({
        'classifier_dataset_hparams': train_dataset.hparams,
    })
    return hparams


def setup_datasets(model_hparams, batch_size, train_dataset, val_dataset, take: Optional[int] = None):
    # Dataset preprocessing
    train_tf_dataset = train_dataset.get_datasets(mode='train', shuffle_files=True)
    val_tf_dataset = val_dataset.get_datasets(mode='val', shuffle_files=True)

    train_tf_dataset = train_tf_dataset.shuffle(model_hparams['shuffle_buffer_size'], reshuffle_each_iteration=True)

    # rospy.logerr_once("NOT BALANCING!")
    train_tf_dataset = train_tf_dataset.balance()
    val_tf_dataset = val_tf_dataset.balance()

    train_tf_dataset = batch_tf_dataset(train_tf_dataset, batch_size, drop_remainder=True)
    val_tf_dataset = batch_tf_dataset(val_tf_dataset, batch_size, drop_remainder=True)

    train_tf_dataset = train_tf_dataset.take(take)
    val_tf_dataset = val_tf_dataset.take(take)

    train_tf_dataset = train_tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    val_tf_dataset = val_tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return train_tf_dataset, val_tf_dataset


def train_main(dataset_dirs: List[pathlib.Path],
               model_hparams: pathlib.Path,
               log: str,
               batch_size: int,
               epochs: int,
               seed: int,
               use_gt_rope: bool,
               checkpoint: Optional[pathlib.Path] = None,
               threshold: Optional[float] = None,
               ensemble_idx: Optional[int] = None,
               old_compat: bool = False,
               take: Optional[int] = None,
               validate: bool = True,
               trials_directory: Optional[pathlib.Path] = None,
               **kwargs):
    model_hparams = load_hjson(model_hparams)
    model_class = link_bot_classifiers.get_model(model_hparams['model_class'])

    # set load_true_states=True when debugging
    train_dataset = ClassifierDatasetLoader(dataset_dirs=dataset_dirs,
                                            load_true_states=True,
                                            use_gt_rope=use_gt_rope,
                                            threshold=threshold,
                                            old_compat=old_compat,
                                            )
    val_dataset = ClassifierDatasetLoader(dataset_dirs=dataset_dirs,
                                          load_true_states=True,
                                          use_gt_rope=use_gt_rope,
                                          threshold=threshold,
                                          old_compat=old_compat,
                                          )

    model_hparams.update(setup_hparams(batch_size, dataset_dirs, seed, train_dataset, use_gt_rope))
    model = model_class(hparams=model_hparams, batch_size=batch_size, scenario=train_dataset.scenario)

    checkpoint_name, trial_path = setup_training_paths(checkpoint, ensemble_idx, log, model_hparams, trials_directory)

    if validate:
        mid_epoch_val_batches = 50
        val_every_n_batches = 100
        save_every_n_minutes = 20
        validate_first = True
    else:
        mid_epoch_val_batches = None
        val_every_n_batches = None
        save_every_n_minutes = None
        validate_first = False

    runner = ModelRunner(model=model,
                         training=True,
                         params=model_hparams,
                         trial_path=trial_path,
                         key_metric=AccuracyCheckpointMetric,
                         checkpoint=checkpoint,
                         mid_epoch_val_batches=mid_epoch_val_batches,
                         val_every_n_batches=val_every_n_batches,
                         save_every_n_minutes=save_every_n_minutes,
                         validate_first=validate_first,
                         batch_metadata=train_dataset.batch_metadata)
    train_tf_dataset, val_tf_dataset = setup_datasets(model_hparams, batch_size, train_dataset, val_dataset, take)

    final_val_metrics = runner.train(train_tf_dataset, val_tf_dataset, num_epochs=epochs)

    return trial_path, final_val_metrics


def eval_main(dataset_dirs: List[pathlib.Path],
              mode: str,
              batch_size: int,
              use_gt_rope: bool,
              threshold: Optional[float] = None,
              old_compat: bool = False,
              take: Optional[int] = None,
              checkpoint: Optional[pathlib.Path] = None,
              trials_directory=pathlib.Path,
              **kwargs):
    ###############
    # Model
    ###############
    trial_path = checkpoint.parent.absolute()
    _, params = filepath_tools.create_or_load_trial(trial_path=trial_path, trials_directory=trials_directory)
    model_class = link_bot_classifiers.get_model(params['model_class'])

    ###############
    # Dataset
    ###############
    dataset = ClassifierDatasetLoader(dataset_dirs,
                                      load_true_states=True,
                                      use_gt_rope=use_gt_rope,
                                      old_compat=old_compat,
                                      threshold=threshold)
    tf_dataset = dataset.get_datasets(mode=mode)
    rospy.loginfo(Fore.CYAN + "NOTE! These metrics are on the balanced dataset")
    tf_dataset = tf_dataset.balance()
    tf_dataset = tf_dataset.take(take)

    ###############
    # Evaluate
    ###############
    tf_dataset = batch_tf_dataset(tf_dataset, batch_size, drop_remainder=True)

    model = model_class(hparams=params, batch_size=batch_size, scenario=dataset.scenario)
    # This call to model runner restores the model
    runner = ModelRunner(model=model,
                         training=False,
                         params=params,
                         checkpoint=checkpoint,
                         trial_path=trial_path,
                         key_metric=AccuracyCheckpointMetric,
                         batch_metadata=dataset.batch_metadata)

    val_metrics = model.create_metrics()
    runner.val_epoch(tf_dataset, val_metrics)
    for metric_name, metric_value in val_metrics.items():
        print(f"{metric_name:30s}: {metric_value.result().numpy().squeeze():.4f}")

    return val_metrics


def viz_main(dataset_dirs: List[pathlib.Path],
             checkpoint: pathlib.Path,
             mode: str,
             batch_size: int,
             start_at: int,
             only_errors: bool,
             only_fp: bool,
             only_fn: bool,
             only_tp: bool,
             only_tn: bool,
             only_negative: bool,
             only_positive: bool,
             use_gt_rope: bool,
             old_compat: bool = False,
             threshold: Optional[float] = None,
             **kwargs):
    stdev_pub_ = rospy.Publisher("stdev", Float32, queue_size=10)
    traj_idx_pub_ = rospy.Publisher("traj_idx_viz", Float32, queue_size=10)

    ###############
    # Model
    ###############
    trials_directory = pathlib.Path('trials').absolute()
    trial_path = checkpoint.parent.absolute()
    _, params = filepath_tools.create_or_load_trial(trial_path=trial_path,
                                                    trials_directory=trials_directory)
    model_class = link_bot_classifiers.get_model(params['model_class'])

    ###############
    # Dataset
    ###############
    dataset = ClassifierDatasetLoader(dataset_dirs,
                                      load_true_states=True,
                                      use_gt_rope=use_gt_rope,
                                      threshold=threshold,
                                      old_compat=old_compat)
    model = model_class(hparams=params, batch_size=batch_size, scenario=dataset.scenario)
    tf_dataset = dataset.get_datasets(mode=mode)
    scenario = dataset.scenario

    ###############
    # Evaluate
    ###############
    tf_dataset = batch_tf_dataset(tf_dataset, batch_size, drop_remainder=True)

    model = classifier_utils.load_generic_model(checkpoint)

    for batch_idx, example in enumerate(progressbar(tf_dataset, widgets=base_dataset.widgets)):
        # for batch_idx, example in enumerate(tf_dataset):

        if batch_idx < start_at:
            continue

        example.update(dataset.batch_metadata)
        predictions, _ = model.check_constraint_from_example(example, training=False)

        labels = tf.expand_dims(example['is_close'][:, 1:], axis=2)

        probabilities = predictions['probabilities']

        # Visualization
        example.pop("time")
        example.pop("batch_size")
        decisions = tf.squeeze(probabilities > 0.5, axis=-1)
        labels = tf.squeeze(tf.cast(labels, tf.bool), axis=-1)
        classifier_is_correct = tf.equal(decisions, labels)
        is_tp = tf.logical_and(labels, decisions)
        is_tn = tf.logical_and(tf.logical_not(labels), tf.logical_not(decisions))
        is_fp = tf.logical_and(tf.logical_not(labels), decisions)
        is_fn = tf.logical_and(labels, tf.logical_not(decisions))
        is_negative = tf.logical_not(labels)
        is_positive = labels
        for b in range(batch_size):
            example_b = index_dict_of_batched_tensors_tf(example, b)

            # if the classifier is correct at all time steps, ignore
            if only_negative:
                if not tf.reduce_all(is_negative[b]):
                    continue
            if only_positive:
                if not tf.reduce_all(is_positive[b]):
                    continue
            if only_tp:
                if not tf.reduce_all(is_tp[b]):
                    continue
            if only_tn:
                if not tf.reduce_all(is_tn[b]):
                    continue
            if only_fp:
                if not tf.reduce_all(is_fp[b]):
                    continue
            if only_fn:
                if not tf.reduce_all(is_fn[b]):
                    continue
            if only_errors:
                if tf.reduce_all(classifier_is_correct[b]):
                    continue

            def _custom_viz_t(scenario: ScenarioWithVisualization, e: Dict, t: int):
                if t > 0:
                    accept_probability_t = predictions['probabilities'][b, t - 1, 0].numpy()
                else:
                    accept_probability_t = -999
                scenario.plot_accept_probability(accept_probability_t)

                traj_idx_msg = Float32()
                traj_idx_msg.data = batch_idx * batch_size + b
                traj_idx_pub_.publish(traj_idx_msg)

            anim = RvizAnimation(scenario=scenario,
                                 n_time_steps=dataset.horizon,
                                 init_funcs=[init_viz_env,
                                             dataset.init_viz_action(),
                                             ],
                                 t_funcs=[_custom_viz_t,
                                          dataset.classifier_transition_viz_t(),
                                          ExperimentScenario.plot_stdev_t,
                                          ])

            with open("debugging.hjson", 'w') as f:
                example_b_np = numpify(example_b)
                my_hdump(example_b_np, f)
            anim.play(example_b)


def eval_ensemble_main(dataset_dir: pathlib.Path,
                       checkpoints: List[pathlib.Path],
                       mode: str,
                       batch_size: int,
                       use_gt_rope: bool,
                       take: Optional[int] = None,
                       balance: Optional[bool] = True,
                       no_plot: Optional[bool] = True,
                       **kwargs):
    ###############
    # Model
    ###############
    models = [load_generic_model(checkpoint) for checkpoint in checkpoints]
    const_keys_for_classifier = []
    ensemble = Ensemble2(models, const_keys_for_classifier)

    ###############
    # Dataset
    ###############
    dataset = ClassifierDatasetLoader([dataset_dir], load_true_states=True, use_gt_rope=use_gt_rope)
    tf_dataset = dataset.get_datasets(mode=mode)
    tf_dataset = tf_dataset.balance()
    tf_dataset = tf_dataset.take(take)
    tf_dataset = tf_dataset.batch(batch_size, drop_remainder=True)

    ###############
    # Evaluate
    ###############
    classifiers_nickname = checkpoints[0].parent.parent.name
    outdir = pathlib.Path('results') / dataset_dir / classifiers_nickname
    outdir.mkdir(exist_ok=True, parents=True)
    outfile = outdir / f'results_{int(time())}.npz'

    labels = []
    classifier_ensemble_stdevs = []
    classifier_is_corrects = []
    classifier_probabilities = []
    for batch_idx, batch in enumerate(progressbar(tf_dataset, widgets=base_dataset.widgets)):
        batch.update(dataset.batch_metadata)

        mean_predictions, stdev_predictions = ensemble(NNClassifierWrapper.check_constraint_from_example, batch)

        mean_probabilities = mean_predictions['probabilities']
        stdev_probabilities = stdev_predictions['probabilities']
        batch_labels = tf.expand_dims(batch['is_close'][:, 1:], axis=2)
        decisions = mean_probabilities > 0.5
        classifier_is_correct = tf.squeeze(tf.equal(decisions, tf.cast(batch_labels, tf.bool)), axis=-1)

        classifier_is_correct = classifier_is_correct.numpy().squeeze()
        classifier_ensemble_stdev = stdev_probabilities.numpy().squeeze()
        mean_probabilities = mean_probabilities.numpy().squeeze()
        batch_labels = batch_labels.numpy().squeeze()

        classifier_ensemble_stdevs.extend(classifier_ensemble_stdev.tolist())
        classifier_is_corrects.extend(classifier_is_correct.tolist())
        classifier_probabilities.extend(mean_probabilities.tolist())
        labels.extend(batch_labels.tolist())

    classifier_ensemble_stdevs = np.array(classifier_ensemble_stdevs)
    classifier_is_corrects = np.array(classifier_is_corrects)
    classifier_probabilities = np.array(classifier_probabilities)
    labels = np.array(labels)
    mean_classifier_ensemble_stdev = tf.math.reduce_mean(classifier_ensemble_stdevs)
    stdev_classifier_ensemble_stdev = tf.math.reduce_std(classifier_ensemble_stdevs)
    datum = {
        'stdevs':        classifier_ensemble_stdevs,
        'is_correct':    classifier_is_corrects,
        'probabilities': classifier_probabilities,
        'labels': labels,
        'checkpoints': [c.as_posix() for c in checkpoints],
        'dataset': dataset_dir.as_posix(),
        'balance': balance,
        'mode': mode,
        'use_gt_rope': use_gt_rope,
    }
    np.savez(outfile, **datum)
    print(f'mean={mean_classifier_ensemble_stdev.numpy()} stdev={stdev_classifier_ensemble_stdev.numpy()}')
    print(f"{outfile.as_posix()}")

    plt.figure()
    ax1 = plt.gca()
    ax1.hist(classifier_ensemble_stdevs, bins=100)
    ax1.set_xlabel("ensemble stdev")
    ax1.set_ylabel("count")

    plt.figure()
    ax2 = plt.gca()
    ax2.violinplot(classifier_ensemble_stdevs)
    ax2.set_xlabel("density")
    ax2.set_ylabel("classifier uncertainty")

    if not no_plot:
        plt.show()
