#!/usr/bin/env python
import pathlib
import pickle
import uuid
from time import time
from typing import List, Optional, Dict, Callable

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from colorama import Fore
from dynamo_pandas.transactions import put_item
from progressbar import progressbar

import link_bot_classifiers
import link_bot_classifiers.get_model
import ros_numpy
import rospy
from analysis.results_utils import try_load_classifier_params
from geometry_msgs.msg import Point
from link_bot_classifiers import classifier_utils
from link_bot_classifiers.base_constraint_checker import classifier_ensemble_check_constraint
from link_bot_classifiers.uncertainty import make_max_class_prob
from link_bot_data import dynamodb_utils
from link_bot_data.classifier_dataset import ClassifierDatasetLoader
from link_bot_data.dataset_utils import batch_tf_dataset, get_filter, deserialize_scene_msg
from link_bot_data.load_dataset import get_classifier_dataset_loader
from link_bot_data.progressbar_widgets import mywidgets
from link_bot_data.visualization import init_viz_env
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from merrrt_visualization.rviz_animation_controller import RvizAnimation
from moonshine import filepath_tools, common_train_hparams
from moonshine.filepath_tools import load_hjson
from moonshine.indexing import index_dict_of_batched_tensors_tf
from moonshine.metrics import AccuracyCheckpointMetric
from moonshine.model_runner import ModelRunner
from moonshine.moonshine_utils import remove_batch
from state_space_dynamics.train_test_dynamics import setup_training_paths
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker


def setup_hparams(batch_size, dataset_dirs, seed, train_dataset, use_gt_rope):
    hparams = common_train_hparams.setup_hparams(batch_size, dataset_dirs, seed, train_dataset)
    hparams.update({
        'classifier_dataset_hparams': train_dataset.hparams,
        'use_gt_rope':                use_gt_rope,
    })
    return hparams


def setup_dataset_loaders(model_hparams,
                          batch_size,
                          train_dataset_loader,
                          val_dataset_loader,
                          seed,
                          take: Optional[int] = None):
    train_dataset = train_dataset_loader.get_datasets(mode='train', shuffle=True)
    val_dataset = val_dataset_loader.get_datasets(mode='val', shuffle=True)

    train_dataset, val_dataset = setup_datasets(model_hparams, batch_size, train_dataset, val_dataset, seed, take)

    return train_dataset, val_dataset


def setup_datasets(model_hparams, batch_size, train_dataset, val_dataset, seed, train_take, val_take=-1):
    if val_take == -1 and train_take is not None:
        val_take = train_take

    if 'shuffle_buffer_size' in model_hparams:
        train_dataset = train_dataset.shuffle(model_hparams['shuffle_buffer_size'],
                                              reshuffle_each_iteration=True,
                                              seed=seed)
    train_dataset = train_dataset.balance()
    train_dataset = train_dataset.take(train_take)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=False)
    val_dataset = val_dataset.balance()
    val_dataset = val_dataset.take(val_take)
    val_dataset = val_dataset.batch(batch_size, drop_remainder=False)
    return train_dataset, val_dataset


def train_main(dataset_dirs: List[pathlib.Path],
               model_hparams: pathlib.Path,
               log: str,
               batch_size: int,
               epochs: int,
               seed: int,
               use_gt_rope: bool = True,
               checkpoint: Optional[pathlib.Path] = None,
               threshold: Optional[float] = None,
               ensemble_idx: Optional[int] = None,
               take: Optional[int] = None,
               no_validate: bool = False,
               trials_directory: Optional[pathlib.Path] = pathlib.Path("./trials").absolute(),
               **kwargs):
    model_hparams = load_hjson(model_hparams)
    model_class = link_bot_classifiers.get_model.get_model(model_hparams['model_class'])

    # set load_true_states=True when debugging
    train_dataset = get_classifier_dataset_loader(dataset_dirs=dataset_dirs,
                                                  load_true_states=True,
                                                  use_gt_rope=use_gt_rope,
                                                  threshold=threshold,
                                                  )
    val_dataset = get_classifier_dataset_loader(dataset_dirs=dataset_dirs,
                                                load_true_states=True,
                                                use_gt_rope=use_gt_rope,
                                                threshold=threshold,
                                                )

    model_hparams.update(setup_hparams(batch_size, dataset_dirs, seed, train_dataset, use_gt_rope))
    if threshold is not None:
        model_hparams['labeling_params']['threshold'] = threshold
    model = model_class(hparams=model_hparams, batch_size=batch_size, scenario=train_dataset.get_scenario())

    checkpoint_name, trial_path = setup_training_paths(checkpoint, log, model_hparams, trials_directory, ensemble_idx)

    if no_validate:
        mid_epoch_val_batches = None
        val_every_n_batches = None
        save_every_n_minutes = None
        validate_first = False
    else:
        mid_epoch_val_batches = 50
        val_every_n_batches = 500
        save_every_n_minutes = 20
        validate_first = True

    runner = ModelRunner(model=model,
                         training=True,
                         params=model_hparams,
                         trial_path=trial_path,
                         key_metric=AccuracyCheckpointMetric,
                         checkpoint=checkpoint,
                         mid_epoch_val_batches=mid_epoch_val_batches,
                         save_every_n_minutes=save_every_n_minutes,
                         validate_first=validate_first,
                         batch_metadata=train_dataset.batch_metadata)
    train_tf_dataset, val_tf_dataset = setup_dataset_loaders(model_hparams, batch_size, train_dataset, val_dataset,
                                                             seed, take)

    final_val_metrics = runner.train(train_tf_dataset, val_tf_dataset, num_epochs=epochs)

    return trial_path, final_val_metrics


def eval_generator(dataset_dirs: List[pathlib.Path],
                   checkpoint: pathlib.Path,
                   mode: str,
                   batch_size: int,
                   use_gt_rope: bool = True,
                   threshold: Optional[float] = None,
                   take: Optional[int] = None,
                   balance: bool = True,
                   scenario: Optional[ScenarioWithVisualization] = None,
                   **kwargs):
    model, runner, tf_dataset = eval_setup(balance=balance,
                                           batch_size=batch_size,
                                           checkpoint=checkpoint,
                                           dataset_dirs=dataset_dirs,
                                           mode=mode,
                                           take=take,
                                           threshold=threshold,
                                           use_gt_rope=use_gt_rope,
                                           scenario=scenario)

    val_metrics = model.create_metrics()
    for example, outputs in runner.val_generator(tf_dataset, val_metrics):
        yield example, outputs


def eval_main(dataset_dirs: pathlib.Path,
              checkpoint: pathlib.Path,
              mode: str,
              batch_size: int,
              use_gt_rope: bool = True,
              threshold: Optional[float] = None,
              take: Optional[int] = None,
              balance: bool = False,
              scenario: Optional[ScenarioWithVisualization] = None,
              profile: Optional[tuple] = None,
              **kwargs):
    model, runner, tf_dataset = eval_setup(balance=balance,
                                           batch_size=batch_size,
                                           checkpoint=checkpoint,
                                           dataset_dirs=dataset_dirs,
                                           mode=mode,
                                           take=take,
                                           threshold=threshold,
                                           use_gt_rope=use_gt_rope,
                                           scenario=scenario,
                                           profile=profile)

    val_metrics = model.create_metrics()
    runner.val_epoch(tf_dataset, val_metrics)
    # for metric_name, metric_value in val_metrics.items():
    #     print(f"{metric_name:30s}: {metric_value.result().numpy().squeeze():.4f}")
    metric_keys_to_print = ['accuracy', 'precision', 'recall', 'accuracy on negatives']
    metrics_to_print = [f"{val_metrics[k].result().numpy().squeeze():.4f}" for k in metric_keys_to_print]
    print("\t".join(metrics_to_print))

    # Upload the results to the database
    put_eval_in_database(val_metrics=val_metrics,
                         balance=balance,
                         batch_size=batch_size,
                         checkpoint=checkpoint,
                         dataset_dirs=dataset_dirs,
                         mode=mode,
                         model=model,
                         profile=profile,
                         take=take,
                         threshold=threshold,
                         use_gt_rope=use_gt_rope,
                         kwargs=kwargs)

    return val_metrics


def eval_n_main(dataset_dir: pathlib.Path,
                checkpoints: List[pathlib.Path],
                mode: str,
                batch_size: int,
                use_gt_rope: bool = True,
                threshold: Optional[float] = None,
                take: Optional[int] = None,
                balance: bool = False,
                scenario: Optional[ScenarioWithVisualization] = None,
                **kwargs):
    dataset_loader, dataset = setup_eval_dataset(balance, [dataset_dir], mode, scenario, take, threshold,
                                                 use_gt_rope, batch_size)

    metric_keys_to_print = ['accuracy', 'precision', 'recall', 'accuracy on negatives']
    all_metrics_to_print = []
    for checkpoint in checkpoints:
        trial_path = checkpoint.parent.absolute()
        _, params = filepath_tools.create_or_load_trial(trial_path=trial_path)
        model_class = link_bot_classifiers.get_model.get_model(params['model_class'])

        model = model_class(hparams=params, batch_size=batch_size, scenario=dataset_loader.get_scenario(), verbose=-1)
        # This call to model runner restores the model
        runner = ModelRunner(model=model,
                             training=False,
                             params=params,
                             checkpoint=checkpoint,
                             trial_path=trial_path,
                             key_metric=AccuracyCheckpointMetric,
                             batch_metadata=dataset_loader.batch_metadata,
                             verbose=-1)

        val_metrics = model.create_metrics()
        runner.val_epoch(dataset, val_metrics)
        metrics_to_print = [f"{val_metrics[k].result().numpy().squeeze():.4f}" for k in metric_keys_to_print]
        all_metrics_to_print.append(metrics_to_print)

        put_eval_in_database(val_metrics=val_metrics,
                             balance=balance,
                             batch_size=batch_size,
                             checkpoint=checkpoint,
                             dataset_dirs=[dataset_dir],
                             mode=mode,
                             model=model,
                             take=take,
                             threshold=threshold,
                             use_gt_rope=use_gt_rope,
                             kwargs=kwargs)

    for metrics_to_print in all_metrics_to_print:
        print("\t".join(metrics_to_print))


def eval_setup(balance,
               batch_size,
               checkpoint,
               dataset_dirs,
               mode,
               take,
               threshold,
               use_gt_rope,
               scenario,
               **kwargs):
    trial_path = checkpoint.parent.absolute()
    _, params = filepath_tools.create_or_load_trial(trial_path=trial_path)
    model_class = link_bot_classifiers.get_model.get_model(params['model_class'])

    dataset_loader, dataset = setup_eval_dataset(balance, dataset_dirs, mode, scenario, take, threshold, use_gt_rope,
                                                 batch_size)

    model = model_class(hparams=params, batch_size=batch_size, scenario=dataset_loader.get_scenario())
    # This call to model runner restores the model
    runner = ModelRunner(model=model,
                         training=False,
                         params=params,
                         checkpoint=checkpoint,
                         trial_path=trial_path,
                         key_metric=AccuracyCheckpointMetric,
                         batch_metadata=dataset_loader.batch_metadata,
                         **kwargs)
    return model, runner, dataset


def put_eval_in_database(val_metrics,
                         balance,
                         batch_size,
                         checkpoint,
                         dataset_dirs,
                         mode,
                         model,
                         profile=None,
                         take=None,
                         threshold=None,
                         use_gt_rope=None,
                         kwargs=None):
    if kwargs is None:
        kwargs = {}

    classifier_hparams = try_load_classifier_params(checkpoint)
    classifier_source_env = classifier_hparams['classifier_dataset_hparams']['scene_name']
    original_training_seed = classifier_hparams['seed']
    fine_tuning_seed = classifier_hparams.get('fine_tuning_seed', None)
    fine_tuning_take = classifier_hparams.get('fine_tuning_take', None)
    fine_tuning_dataset_dirs = classifier_hparams.get('fine_tuning_dataset_dirs', None)
    fine_tune_conv = classifier_hparams.get('fine_tune_conv', None)
    fine_tune_lstm = classifier_hparams.get('fine_tune_lstm', None)
    fine_tune_dense = classifier_hparams.get('fine_tune_dense', None)
    fine_tune_output = classifier_hparams.get('fine_tune_output', None)
    learning_rate = classifier_hparams.get('learning_rate', None)

    item = {
        'uuid':                     str(uuid.uuid4()),
        'time':                     str(int(time())),
        'classifier':               checkpoint.as_posix(),
        'dataset_dirs':             ','.join([d.as_posix() for d in dataset_dirs]),
        'balance':                  balance,
        'threshold':                threshold,
        'use_gt_rope':              use_gt_rope,
        'batch_size':               batch_size,
        'mode':                     mode,
        'take':                     take,
        'profile':                  str(profile),
        'include_robot_geometry':   getattr(model.hparams, 'include_robot_geometry', None),
        'do_augmentation':          model.aug.do_augmentation(),
        'classifier_source_env':    classifier_source_env,
        'original_training_seed':   original_training_seed,
        'fine_tuning_seed':         fine_tuning_seed,
        'fine_tuning_take':         fine_tuning_take,
        'learning_rate':            learning_rate,
        'fine_tuning_dataset_dirs': fine_tuning_dataset_dirs,
        'fine_tune_conv':           fine_tune_conv,
        'fine_tune_lstm':           fine_tune_lstm,
        'fine_tune_dense':          fine_tune_dense,
        'fine_tune_output':         fine_tune_output,
    }
    item.update({k: float(v.result().numpy().squeeze()) for k, v in val_metrics.items()})
    put_item(item=item, table=dynamodb_utils.classifier_table(kwargs.get("debug", False)))


def compare_main(dataset_dirs: List[pathlib.Path],
                 checkpoint1: pathlib.Path,
                 checkpoint2: pathlib.Path,
                 mode: str,
                 batch_size: int,
                 use_gt_rope: bool = True,
                 threshold: Optional[float] = None,
                 take: Optional[int] = None,
                 balance: bool = True,
                 scenario: Optional[ScenarioWithVisualization] = None,
                 **kwargs):
    dataset, tf_dataset = setup_eval_dataset(balance, dataset_dirs, mode, scenario, take, threshold, use_gt_rope,
                                             batch_size)

    model1 = classifier_utils.load_generic_model(checkpoint1)
    model2 = classifier_utils.load_generic_model(checkpoint2)

    for inputs in tf_dataset:
        inputs.update(dataset.batch_metadata)
        predictions1 = model1.check_constraint_from_example(inputs, training=False)
        predictions2 = model2.check_constraint_from_example(inputs, training=False)

        inputs.pop("batch_size")
        inputs.pop("time")
        inputs.pop("kinect_params")

        for b in range(batch_size):
            p1 = remove_batch(predictions1['probabilities'][b])[0]
            p2 = remove_batch(predictions2['probabilities'][b])[0]
            p1_decision = p1 > 0.5
            p2_decision = p2 > 0.5
            if p1_decision != p2_decision:
                # visualize!
                anim = RvizAnimation(scenario=dataset.get_scenario(),
                                     n_time_steps=dataset.horizon,
                                     init_funcs=[init_viz_env,
                                                 dataset.init_viz_action(),
                                                 ],
                                     t_funcs=[init_viz_env,
                                              dataset.classifier_transition_viz_t(),
                                              ])

                deserialize_scene_msg(inputs)
                inputs_b = index_dict_of_batched_tensors_tf(inputs, b)
                anim.play(inputs_b)


def setup_eval_dataset(balance, dataset_dirs, mode, scenario, take, threshold, use_gt_rope, batch_size):
    dataset_loader = get_classifier_dataset_loader(dataset_dirs,
                                                   load_true_states=True,
                                                   use_gt_rope=use_gt_rope,
                                                   threshold=threshold,
                                                   scenario=scenario)
    dataset = dataset_loader.get_datasets(mode=mode)
    if balance:
        rospy.loginfo(Fore.CYAN + "NOTE! These metrics are on the balanced dataset")
        dataset = dataset.balance()
    dataset = dataset.take(take)
    dataset = batch_tf_dataset(dataset, batch_size, drop_remainder=False)
    return dataset_loader, dataset


class ClassifierEvaluation:
    def __init__(self, dataset_dirs: List[pathlib.Path],
                 checkpoint: pathlib.Path,
                 mode: str,
                 batch_size: int,
                 start_at: int,
                 use_gt_rope: bool = True,
                 take: int = None,
                 threshold: Optional[float] = None,
                 **kwargs):
        self.start_at = start_at
        trials_directory = pathlib.Path('trials').absolute()
        trial_path = checkpoint.parent.absolute()
        _, params = filepath_tools.create_or_load_trial(trial_path=trial_path,
                                                        trials_directory=trials_directory)

        # Dataset
        self.dataset_loader = get_classifier_dataset_loader(dataset_dirs,
                                                            load_true_states=True,
                                                            use_gt_rope=use_gt_rope,
                                                            threshold=threshold)
        self.dataset = self.dataset_loader.get_datasets(mode=mode)

        # Iterate
        self.dataset = self.dataset.batch(batch_size, drop_remainder=False)
        if take is not None:
            self.dataset = self.dataset.take(take)

        self.model = classifier_utils.load_generic_model(checkpoint)
        self.scenario = self.dataset_loader.get_scenario()

    def __iter__(self):
        for batch_idx, example in enumerate(progressbar(self.dataset, widgets=mywidgets)):
            if batch_idx < self.start_at:
                continue

            example.update(self.dataset_loader.batch_metadata)
            predictions = self.model.check_constraint_from_example(example, training=False)

            yield batch_idx, example, predictions


class ClassifierEvaluationFilter:
    def __init__(self, dataset_dirs: List[pathlib.Path],
                 checkpoint: pathlib.Path,
                 mode: str,
                 should_keep_example: Callable,
                 start_at: int = 0,
                 use_gt_rope: bool = True,
                 take: int = None,
                 take_after_filter: int = None,
                 threshold: Optional[float] = None,
                 **kwargs):
        self.view = ClassifierEvaluation(dataset_dirs=dataset_dirs,
                                         checkpoint=checkpoint,
                                         mode=mode,
                                         batch_size=1,
                                         start_at=start_at,
                                         use_gt_rope=use_gt_rope,
                                         take=take,
                                         threshold=threshold,
                                         **kwargs)
        self.take_after_filter = take_after_filter
        self.should_keep_example = should_keep_example
        self.scenario = self.view.scenario
        self.dataset = self.view.dataset_loader
        self.model = self.view.model

    def __iter__(self):
        count = 0
        for batch_idx, example, predictions in self.view:
            if self.take_after_filter is not None and count >= self.take_after_filter:
                return

            if self.should_keep_example(remove_batch(example), predictions):
                yield batch_idx, example, predictions
                count += 1


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
             use_gt_rope: bool = True,
             threshold: Optional[float] = None,
             **kwargs):
    count = 0

    view = ClassifierEvaluation(dataset_dirs=dataset_dirs,
                                checkpoint=checkpoint,
                                mode=mode,
                                batch_size=batch_size,
                                start_at=start_at,
                                use_gt_rope=use_gt_rope,
                                threshold=threshold,
                                **kwargs)

    for batch_idx, example, predictions in view:
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

            count += 1

            def _custom_viz_t(scenario: ScenarioWithVisualization, e: Dict, t: int):
                if t > 0:
                    accept_probability_t = predictions['probabilities'][b, t - 1, 0].numpy()
                else:
                    accept_probability_t = -999
                scenario.plot_accept_probability(accept_probability_t)
                scenario.plot_traj_idx_rviz(batch_idx * batch_size + b)

            anim = RvizAnimation(scenario=view.dataset_loader.get_scenario(),
                                 n_time_steps=view.dataset_loader.horizon,
                                 init_funcs=[init_viz_env,
                                             view.dataset_loader.init_viz_action(),
                                             ],
                                 t_funcs=[_custom_viz_t,
                                          init_viz_env,
                                          view.dataset_loader.classifier_transition_viz_t(),
                                          ExperimentScenario.plot_dynamics_stdev_t,
                                          ])

            deserialize_scene_msg(example_b)
            with open("debugging.pkl", 'wb') as f:
                pickle.dump(example_b, f)
            anim.play(example_b)

    print(count)


def run_ensemble_on_dataset(dataset_dir: pathlib.Path,
                            ensemble_path: pathlib.Path,
                            mode: str,
                            batch_size: int,
                            use_gt_rope: bool = True,
                            take: Optional[int] = None,
                            balance: Optional[bool] = True,
                            **kwargs):
    ensemble = classifier_utils.load_generic_model(ensemble_path)

    # Dataset
    dataset = ClassifierDatasetLoader([dataset_dir], load_true_states=True, use_gt_rope=use_gt_rope)
    tf_dataset = dataset.get_datasets(mode=mode)
    if balance:
        tf_dataset = tf_dataset.balance()
    tf_dataset = tf_dataset.take(take)
    tf_dataset = tf_dataset.batch(batch_size, drop_remainder=False)

    # Evaluate
    for batch_idx, batch in enumerate(progressbar(tf_dataset, widgets=mywidgets)):
        batch.update(dataset.batch_metadata)

        mean_predictions, stdev_predictions = ensemble.check_constraint_from_example(batch)

        yield dataset, batch_idx, batch, mean_predictions, stdev_predictions


def eval_ensemble_main(dataset_dir: pathlib.Path,
                       ensemble_path: pathlib.Path,
                       mode: str,
                       batch_size: int,
                       use_gt_rope: bool = True,
                       take: Optional[int] = None,
                       balance: Optional[bool] = True,
                       no_plot: Optional[bool] = True,
                       **kwargs):
    classifiers_nickname = ensemble_path.parent.name
    outdir = pathlib.Path('results') / dataset_dir / classifiers_nickname
    outdir.mkdir(exist_ok=True, parents=True)
    outfile = outdir / f'results_{int(time())}.npz'

    labels = []
    errors = []
    classifier_ensemble_stdevs = []
    classifier_is_corrects = []
    classifier_probabilities = []

    itr = run_ensemble_on_dataset(dataset_dir=dataset_dir,
                                  ensemble_path=ensemble_path,
                                  mode=mode,
                                  batch_size=batch_size,
                                  use_gt_rope=use_gt_rope,
                                  take=take,
                                  balance=balance,
                                  **kwargs)
    for dataset, batch_idx, batch, mean_predictions, stdev_predictions in itr:
        mean_probabilities = mean_predictions['probabilities']
        stdev_probabilities = stdev_predictions['probabilities']
        batch_labels = tf.expand_dims(batch['is_close'][:, 1:], axis=2)
        batch_error = batch['error'][:, 1]
        decisions = mean_probabilities > 0.5
        classifier_is_correct = tf.squeeze(tf.equal(decisions, tf.cast(batch_labels, tf.bool)), axis=-1)

        classifier_is_correct = classifier_is_correct.numpy().squeeze()
        classifier_ensemble_stdev = stdev_probabilities.numpy().squeeze()
        mean_probabilities = mean_probabilities.numpy().squeeze()
        batch_labels = batch_labels.numpy().squeeze()
        batch_error = batch_error.numpy().squeeze()

        classifier_ensemble_stdevs.extend(classifier_ensemble_stdev.tolist())
        classifier_is_corrects.extend(classifier_is_correct.tolist())
        classifier_probabilities.extend(mean_probabilities.tolist())
        labels.extend(batch_labels.tolist())
        errors.extend(batch_error.tolist())

    classifier_ensemble_stdevs = np.array(classifier_ensemble_stdevs)
    classifier_is_corrects = np.array(classifier_is_corrects)
    classifier_probabilities = np.array(classifier_probabilities)
    labels = np.array(labels)
    errors = np.array(errors)
    mean_classifier_ensemble_stdev = tf.math.reduce_mean(classifier_ensemble_stdevs)
    stdev_classifier_ensemble_stdev = tf.math.reduce_std(classifier_ensemble_stdevs)
    datum = {
        'stdevs':        classifier_ensemble_stdevs,
        'is_correct':    classifier_is_corrects,
        'probabilities': classifier_probabilities,
        'labels':        labels,
        'ensemble_path': ensemble_path.as_posix(),
        'dataset':       dataset_dir.as_posix(),
        'balance':       balance,
        'error':         errors,
        'mode':          mode,
        'use_gt_rope':   use_gt_rope,
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


def viz_ensemble_main(dataset_dir: pathlib.Path,
                      ensemble_path: pathlib.Path,
                      mode: str,
                      batch_size: int,
                      use_gt_rope: bool = True,
                      take: Optional[int] = None,
                      balance: Optional[bool] = True,
                      **kwargs):
    grippers_pub = rospy.Publisher("grippers_viz_pub", MarkerArray, queue_size=10)

    itr = run_ensemble_on_dataset(dataset_dir=dataset_dir,
                                  ensemble_path=ensemble_path,
                                  mode=mode,
                                  batch_size=batch_size,
                                  use_gt_rope=use_gt_rope,
                                  take=take,
                                  balance=balance,
                                  **kwargs)
    # TODO: use ClassifierEvaluationFilter

    for dataset, batch_idx, batch, mean_predictions, stdev_predictions in itr:
        mean_probabilities = mean_predictions['probabilities']
        mean_mcps = make_max_class_prob(mean_probabilities)
        stdev_probabilities = stdev_predictions['probabilities']
        batch_labels = tf.expand_dims(batch['is_close'][:, 1:], axis=2)
        decisions = mean_probabilities > 0.5
        is_correct = tf.squeeze(tf.equal(decisions, tf.cast(batch_labels, tf.bool)), axis=-1)

        is_correct = is_correct.numpy().squeeze()
        ensemble_stdev = stdev_probabilities.numpy().squeeze()
        ensemble_mean = mean_probabilities.numpy().squeeze()
        batch_labels = batch_labels.numpy().squeeze()

        deserialize_scene_msg(batch)
        batch.pop('batch_size')
        for k in dataset.batch_metadata.keys():
            batch.pop(k)

        for b in range(batch_size):
            example_b = index_dict_of_batched_tensors_tf(batch, b)
            ensemble_mcp_b = mean_mcps[b]
            is_correct_b = is_correct[b]
            ensemble_stdev_b = ensemble_stdev[b]
            ensemble_mean_b = ensemble_mean[b]
            label_b = batch_labels[b]
            example_b['accept_probability'] = [ensemble_mean_b]

            stdev_filter = get_filter('stdev', **kwargs)
            mcp_filter = get_filter('mcp', **kwargs)
            label_filter = get_filter('label', **kwargs)

            A = tf.constant([[0.20057761, 2.24315289]])
            B = tf.constant([-1.29731397])
            accept = classifier_ensemble_check_constraint(A, B, tf.expand_dims(ensemble_mean_b, axis=0),
                                                          tf.expand_dims(ensemble_stdev_b, axis=0))

            decision_b = ensemble_mean_b > 0.5
            is_fp = tf.logical_and(tf.logical_not(label_b), decision_b)
            if not is_fp:
                continue

            if not stdev_filter(ensemble_stdev_b):
                continue

            if not mcp_filter(ensemble_mcp_b):
                continue

            if not label_filter(label_b):
                continue

            print(ensemble_mean_b, ensemble_stdev_b, bool(accept.numpy().squeeze()))

            marker = Marker()
            marker.ns = 'grippers'
            marker.id = batch_idx * batch_size + b
            marker.color = ColorRGBA(0.5, 0.5, 0.5, 0.5)
            marker.header.frame_id = 'world'
            marker.header.stamp = rospy.Time.now()
            marker.action = Marker.ADD
            marker.type = Marker.LINE_LIST
            marker.scale.x = 0.02
            marker.points = [ros_numpy.msgify(Point, example_b['left_gripper'][0]),
                             ros_numpy.msgify(Point, example_b['right_gripper'][0])]
            msg = MarkerArray(markers=[marker])
            grippers_pub.publish(msg)

            def _custom_viz_t(scenario: ScenarioWithVisualization, e: Dict, t: int):
                pass

            anim = RvizAnimation(scenario=dataset.get_scenario(),
                                 n_time_steps=dataset.horizon,
                                 init_funcs=[init_viz_env,
                                             dataset.init_viz_action(),
                                             ],
                                 t_funcs=[_custom_viz_t,
                                          ExperimentScenario.plot_accept_probability_t,
                                          dataset.classifier_transition_viz_t(),
                                          ExperimentScenario.plot_dynamics_stdev_t,
                                          ])

            anim.play(example_b)
