#!/usr/bin/env python
import pathlib
from typing import List, Optional, Dict

import hjson
import numpy as np
import tensorflow as tf
from progressbar import progressbar

import link_bot_classifiers
import rospy
from link_bot_classifiers import classifier_utils
from link_bot_classifiers.classifier_utils import load_generic_model
from link_bot_data import base_dataset
from link_bot_data.balance import balance
from link_bot_data.classifier_dataset import ClassifierDatasetLoader
from link_bot_data.dataset_utils import add_predicted, batch_tf_dataset
from link_bot_pycommon.base_3d_scenario import Base3DScenario
from link_bot_pycommon.collision_checking import batch_in_collision_tf_3d
from moonshine.indexing import index_dict_of_batched_tensors_tf
from moonshine.metric import AccuracyMetric
from shape_completion_training.model import filepath_tools
from shape_completion_training.model_runner import ModelRunner
from state_space_dynamics import common_train_hparams
from state_space_dynamics.train_test import setup_training_paths
from std_msgs.msg import Float32


def setup_hparams(batch_size, dataset_dirs, seed, train_dataset, use_gt_rope):
    hparams = common_train_hparams.setup_hparams(batch_size, dataset_dirs, seed, train_dataset, use_gt_rope)
    hparams.update({
        'classifier_dataset_hparams': train_dataset.hparams,
    })
    return hparams


def setup_datasets(model_hparams, batch_size, seed, train_dataset, val_dataset, take):
    # Dataset preprocessing
    train_tf_dataset = train_dataset.get_datasets(mode='train', shuffle_files=True, take=take)
    val_tf_dataset = val_dataset.get_datasets(mode='val', shuffle_files=True, take=take)

    train_tf_dataset = train_tf_dataset.shuffle(model_hparams['shuffle_buffer_size'], reshuffle_each_iteration=True)

    # rospy.logerr_once("NOT BALANCING!")
    train_tf_dataset = balance(train_tf_dataset)
    val_tf_dataset = balance(val_tf_dataset)

    train_tf_dataset = batch_tf_dataset(train_tf_dataset, batch_size, drop_remainder=True)
    val_tf_dataset = batch_tf_dataset(val_tf_dataset, batch_size, drop_remainder=True)

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
    model_hparams = hjson.load(model_hparams.open('r'))
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
        mid_epoch_val_batches = 100
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
                         key_metric=AccuracyMetric,
                         checkpoint=checkpoint,
                         mid_epoch_val_batches=mid_epoch_val_batches,
                         val_every_n_batches=val_every_n_batches,
                         save_every_n_minutes=save_every_n_minutes,
                         validate_first=validate_first,
                         batch_metadata=train_dataset.batch_metadata)
    train_tf_dataset, val_tf_dataset = setup_datasets(model_hparams, batch_size, seed, train_dataset, val_dataset, take)

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
    tf_dataset = dataset.get_datasets(mode=mode, take=take)
    tf_dataset = balance(tf_dataset)

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
                         key_metric=AccuracyMetric,
                         batch_metadata=dataset.batch_metadata)

    metrics = runner.val_epoch(tf_dataset)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name:30s}: {metric_value}")
    return metrics


def viz_main(dataset_dirs: List[pathlib.Path],
             checkpoint: pathlib.Path,
             mode: str,
             batch_size: int,
             only_errors: bool,
             only_fp: bool,
             only_fn: bool,
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

    model = classifier_utils.load_generic_model([checkpoint])

    fn = 0
    fp = 0
    for batch_idx, example in enumerate(progressbar(tf_dataset, widgets=base_dataset.widgets)):
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
        is_fp = tf.logical_and(tf.logical_not(labels), decisions)
        is_fn = tf.logical_and(labels, tf.logical_not(decisions))
        for b in range(batch_size):
            example_b = index_dict_of_batched_tensors_tf(example, b)

            if tf.reduce_all(is_fp[b]):
                fp += 1
            if tf.reduce_all(is_fn[b]):
                fn += 1

            # if the classifier is correct at all time steps, ignore
            if only_fp:
                if not tf.reduce_all(is_fp[b]):
                    continue
                else:
                    print(f'\nFP: example={batch_idx}, b={b}')
            if only_fn:
                if not tf.reduce_all(is_fn[b]):
                    continue
                else:
                    print(f'\nFN: example={batch_idx}, b={b}')
            if only_errors:
                if tf.reduce_all(classifier_is_correct[b]):
                    continue
                else:
                    if tf.reduce_all(is_fp[b]):
                        print(f'\nFP: example={batch_idx}, b={b}')
                    if tf.reduce_all(is_fn[b]):
                        print(f'\nFN: example={batch_idx}, b={b}')

            def _custom_viz_t(scenario: Base3DScenario, e: Dict, t: int):
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
            # print(fn / (fn + fp), fp / (fn + fp))

    print(fn / (fn + fp), fp / (fn + fp))


def viz_ensemble_main(dataset_dir: pathlib.Path,
                      checkpoints: List[pathlib.Path],
                      mode: str,
                      batch_size: int,
                      only_errors: bool,
                      use_gt_rope: bool,
                      **kwargs):
    dynamics_stdev_pub_ = rospy.Publisher("dynamics_stdev", Float32, queue_size=10)
    classifier_stdev_pub_ = rospy.Publisher("classifier_stdev", Float32, queue_size=10)
    accept_probability_pub_ = rospy.Publisher("accept_probability_viz", Float32, queue_size=10)
    traj_idx_pub_ = rospy.Publisher("traj_idx_viz", Float32, queue_size=10)

    ###############
    # Model
    ###############
    model = load_generic_model(checkpoints)

    ###############
    # Dataset
    ###############
    test_dataset = ClassifierDatasetLoader([dataset_dir], load_true_states=True, use_gt_rope=use_gt_rope)
    test_tf_dataset = test_dataset.get_datasets(mode=mode)
    test_tf_dataset = batch_tf_dataset(test_tf_dataset, batch_size, drop_remainder=True)
    scenario = test_dataset.scenario

    ###############
    # Evaluate
    ###############

    # Iterate over test set
    all_accuracies_over_time = []
    all_stdevs = []
    all_labels = []
    classifier_ensemble_stdevs = []
    for batch_idx, test_batch in enumerate(test_tf_dataset):
        test_batch.update(test_dataset.batch_metadata)

        mean_predictions, stdev_predictions = model.check_constraint_from_example(test_batch)
        mean_probabilities = mean_predictions['probabilities']
        stdev_probabilities = stdev_predictions['probabilities']

        labels = tf.expand_dims(test_batch['is_close'][:, 1:], axis=2)

        all_labels = tf.concat((all_labels, tf.reshape(test_batch['is_close'][:, 1:], [-1])), axis=0)
        all_stdevs = tf.concat((all_stdevs, tf.reshape(test_batch[add_predicted('stdev')], [-1])), axis=0)

        accuracy_over_time = tf.keras.metrics.binary_accuracy(y_true=labels, y_pred=mean_probabilities)
        all_accuracies_over_time.append(accuracy_over_time)

        # Visualization
        test_batch.pop("time")
        test_batch.pop("batch_size")
        decisions = mean_probabilities > 0.5
        classifier_is_correct = tf.squeeze(tf.equal(decisions, tf.cast(labels, tf.bool)), axis=-1)
        for b in range(batch_size):
            example = index_dict_of_batched_tensors_tf(test_batch, b)

            classifier_ensemble_stdev = stdev_probabilities[b].numpy().squeeze()
            classifier_ensemble_stdevs.append(classifier_ensemble_stdev)

            # if the classifier is correct at all time steps, ignore
            if only_errors and tf.reduce_all(classifier_is_correct[b]):
                print(f'batch {b}')
                continue

            # if only_collision
            predicted_rope_states = tf.reshape(example[add_predicted('link_bot')][1], [-1, 3])
            xs = predicted_rope_states[:, 0]
            ys = predicted_rope_states[:, 1]
            zs = predicted_rope_states[:, 2]
            in_collision = bool(batch_in_collision_tf_3d(environment=example,
                                                         xs=xs, ys=ys, zs=zs,
                                                         inflate_radius_m=0)[0].numpy())
            label = bool(example['is_close'][1].numpy())
            accept = decisions[b, 0, 0].numpy()
            # if not (in_collision and accept):
            #     continue

            scenario.plot_environment_rviz(example)

            stdev_probabilities[b].numpy().squeeze()
            classifier_stdev_msg = Float32()
            classifier_stdev_msg.data = stdev_probabilities[b].numpy().squeeze()
            classifier_stdev_pub_.publish(classifier_stdev_msg)

            actual_0 = scenario.index_state_time(example, 0)
            actual_1 = scenario.index_state_time(example, 1)
            pred_0 = scenario.index_predicted_state_time(example, 0)
            pred_1 = scenario.index_predicted_state_time(example, 1)
            action = scenario.index_action_time(example, 0)
            label = example['is_close'][1]
            scenario.plot_state_rviz(actual_0, label='actual', color='#FF0000AA', idx=0)
            scenario.plot_state_rviz(actual_1, label='actual', color='#E00016AA', idx=1)
            scenario.plot_state_rviz(pred_0, label='predicted', color='#0000FFAA', idx=0)
            scenario.plot_state_rviz(pred_1, label='predicted', color='#0553FAAA', idx=1)
            scenario.plot_action_rviz(pred_0, action)
            scenario.plot_is_close(label)

            dynamics_stdev_t = example[add_predicted('stdev')][1, 0].numpy()
            dynamics_stdev_msg = Float32()
            dynamics_stdev_msg.data = dynamics_stdev_t
            dynamics_stdev_pub_.publish(dynamics_stdev_msg)

            accept_probability_t = mean_probabilities[b, 0, 0].numpy()
            accept_probability_msg = Float32()
            accept_probability_msg.data = accept_probability_t
            accept_probability_pub_.publish(accept_probability_msg)

            traj_idx_msg = Float32()
            traj_idx_msg.data = batch_idx * batch_size + b
            traj_idx_pub_.publish(traj_idx_msg)

            # stepper = RvizSimpleStepper()
            # stepper.step()

        print(np.mean(classifier_ensemble_stdevs))

    all_accuracies_over_time = tf.concat(all_accuracies_over_time, axis=0)
    mean_accuracies_over_time = tf.reduce_mean(all_accuracies_over_time, axis=0)
    std_accuracies_over_time = tf.math.reduce_std(all_accuracies_over_time, axis=0)
    mean_classifier_ensemble_stdev = tf.reduce_mean(classifier_ensemble_stdevs)
    print(mean_classifier_ensemble_stdev)
