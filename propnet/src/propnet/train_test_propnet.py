#!/usr/bin/env python
import pathlib
import pickle
from typing import List, Optional, Dict, Callable

import tensorflow as tf
from colorama import Fore
from tqdm import tqdm

import link_bot_classifiers
import rospy
from link_bot_classifiers import classifier_utils
from link_bot_data.dataset_utils import batch_tf_dataset, deserialize_scene_msg
from link_bot_data.load_dataset import get_dynamics_dataset_loader
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


def setup_dataset_loaders(model_hparams,
                          batch_size,
                          train_dataset_loader,
                          val_dataset_loader,
                          seed,
                          take: Optional[int] = None):
    train_dataset = train_dataset_loader.get_datasets(mode='train', shuffle=True)
    val_dataset = val_dataset_loader.get_datasets(mode='val', shuffle=True)

    if 'shuffle_buffer_size' in model_hparams:
        train_dataset = train_dataset.shuffle(model_hparams['shuffle_buffer_size'],
                                              reshuffle_each_iteration=True,
                                              seed=seed)
    train_dataset = train_dataset.take(take)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=False)
    val_dataset = val_dataset.take(take)
    val_dataset = val_dataset.batch(batch_size, drop_remainder=False)

    return train_dataset, val_dataset


def train_main(dataset_dirs: List[pathlib.Path],
               model_hparams: pathlib.Path,
               log: str,
               batch_size: int,
               epochs: int,
               seed: int,
               checkpoint: Optional[pathlib.Path] = None,
               take: Optional[int] = None,
               no_validate: bool = False,
               trials_directory: Optional[pathlib.Path] = pathlib.Path("./trials").absolute(),
               **kwargs):
    model_hparams = load_hjson(model_hparams)

    train_dataset_loader = get_dynamics_dataset_loader(dataset_dirs)
    val_dataset_loader = get_dynamics_dataset_loader(dataset_dirs)

    model_hparams.update(common_train_hparams.setup_hparams(batch_size, dataset_dirs, seed, train_dataset_loader))

    model = PropNet(hparams=model_hparams, batch_size=batch_size, scenario=train_dataset_loader.get_scenario())

    trial_path = setup_training_paths(checkpoint, log, model_hparams, trials_directory)

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
                         val_every_n_batches=val_every_n_batches,
                         train_batch_metadata=train_dataset_loader.batch_metadata,
                         val_batch_metadata=val_dataset_loader.batch_metadata)
    train_dataset, val_dataset = setup_dataset_loaders(model_hparams,
                                                       batch_size,
                                                       train_dataset_loader,
                                                       val_dataset_loader,
                                                       seed,
                                                       take)

    final_val_metrics = runner.train(train_dataset, val_dataset, num_epochs=epochs)

    return trial_path, final_val_metrics


def eval_generator(dataset_dirs: List[pathlib.Path],
                   checkpoint: pathlib.Path,
                   mode: str,
                   batch_size: int,
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
                                           scenario=scenario,
                                           **kwargs)

    val_metrics = model.create_metrics()
    for example, outputs in runner.val_generator(tf_dataset, val_metrics):
        yield example, outputs


def eval_main(dataset_dirs: pathlib.Path,
              checkpoint: pathlib.Path,
              mode: str,
              batch_size: int,
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
                                           scenario=scenario,
                                           profile=profile)

    val_metrics = model.create_metrics()
    runner.val_epoch(tf_dataset, val_metrics)
    for metric_name, metric_value in val_metrics.items():
        print(f"{metric_name:30s}: {metric_value.result().numpy().squeeze():.4f}")

    return val_metrics


def eval_setup(balance,
               batch_size,
               checkpoint,
               dataset_dirs,
               mode,
               take,
               scenario,
               **kwargs):
    trial_path = checkpoint.parent.absolute()
    _, params = filepath_tools.create_or_load_trial(trial_path=trial_path)
    model_class = link_bot_classifiers.get_model.get_model(params['model_class'])

    dataset_loader, dataset = setup_eval_dataset(scenario=scenario, dataset_dirs=dataset_dirs, mode=mode,
                                                 balance=balance, take=take, batch_size=batch_size)

    model = model_class(hparams=params, batch_size=batch_size, scenario=dataset_loader.get_scenario())
    # This call to model runner restores the model
    runner = ModelRunner(model=model,
                         training=False,
                         params=params,
                         checkpoint=checkpoint,
                         trial_path=trial_path,
                         key_metric=AccuracyCheckpointMetric,
                         train_batch_metadata=dataset_loader.batch_metadata,
                         val_batch_metadata=dataset_loader.batch_metadata,
                         **kwargs)
    return model, runner, dataset


def setup_eval_dataset(scenario, dataset_dirs, mode, balance, take, batch_size):
    dataset_loader = get_dynamics_dataset_loader(dataset_dirs)
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
                 take: int = None,
                 show_progressbar: Optional[bool] = True,
                 **kwargs):
        self.show_progressbar = show_progressbar
        self.start_at = start_at
        trials_directory = pathlib.Path('trials').absolute()
        trial_path = checkpoint.parent.absolute()
        _, params = filepath_tools.create_or_load_trial(trial_path=trial_path, trials_directory=trials_directory)

        # Dataset
        if 'dataset_loader' in kwargs:
            self.dataset_loader = kwargs["dataset_loader"]
        else:
            self.dataset_loader = get_dynamics_dataset_loader(dataset_dirs)
        if 'dataset' in kwargs:
            self.dataset = kwargs["dataset"]
        else:
            self.dataset = self.dataset_loader.get_datasets(mode=mode)

        # Iterate
        self.dataset = self.dataset.batch(batch_size, drop_remainder=False)
        if take is not None:
            self.dataset = self.dataset.take(take)

        self.model = classifier_utils.load_generic_model(checkpoint)
        self.scenario = self.dataset_loader.get_scenario()

    def __iter__(self):
        if self.show_progressbar:
            gen = tqdm(self.dataset)
        else:
            gen = self.dataset
        for batch_idx, example in enumerate(gen):
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
                 take: int = None,
                 take_after_filter: int = None,
                 **kwargs):
        self.view = ClassifierEvaluation(dataset_dirs=dataset_dirs,
                                         checkpoint=checkpoint,
                                         mode=mode,
                                         batch_size=1,
                                         start_at=start_at,
                                         take=take,
                                         **kwargs)
        self.take_after_filter = take_after_filter
        self.should_keep_example = should_keep_example
        self.scenario = self.view.scenario
        self.dataset_loader = self.view.dataset_loader
        self.model = self.view.model

    def __iter__(self):
        count = 0
        for batch_idx, example, predictions in self.view:
            if self.take_after_filter is not None and count >= self.take_after_filter:
                return

            if self.should_keep_example(remove_batch(example), remove_batch(predictions)):
                yield batch_idx, example, predictions
                count += 1


def viz_main(dataset_dirs: List[pathlib.Path],
             checkpoint: pathlib.Path,
             mode: str,
             batch_size: int,
             only_mistakes: bool = False,
             only_fp: bool = False,
             only_fn: bool = False,
             only_tp: bool = False,
             only_tn: bool = False,
             only_negative: bool = False,
             only_positive: bool = False,
             **kwargs):
    count = 0

    def _should_keep_example(example, prediction):
        labels = example['is_close'][1]
        probabilities = prediction['probabilities'][0][0]
        decisions = probabilities > 0.5
        labels = tf.cast(labels, tf.bool)
        classifier_is_correct = tf.equal(decisions, labels)
        is_tp = tf.logical_and(labels, decisions)
        is_tn = tf.logical_and(tf.logical_not(labels), tf.logical_not(decisions))
        is_fp = tf.logical_and(tf.logical_not(labels), decisions)
        is_fn = tf.logical_and(labels, tf.logical_not(decisions))
        is_negative = tf.logical_not(labels)
        is_positive = labels

        # if the classifier is correct at all time steps, ignore
        if only_negative:
            if not tf.reduce_all(is_negative):
                return False
        if only_positive:
            if not tf.reduce_all(is_positive):
                return False
        if only_tp:
            if not tf.reduce_all(is_tp):
                return False
        if only_tn:
            if not tf.reduce_all(is_tn):
                return False
        if only_fp:
            if not tf.reduce_all(is_fp):
                return False
        if only_fn:
            if not tf.reduce_all(is_fn):
                return False
        if only_mistakes:
            if tf.reduce_all(classifier_is_correct):
                return False
        return True

    view = ClassifierEvaluationFilter(dataset_dirs=dataset_dirs,
                                      checkpoint=checkpoint,
                                      mode=mode,
                                      should_keep_example=_should_keep_example,
                                      show_progressbar=False)

    for batch_idx, example, predictions in view:
        # Visualization
        example.pop("time")
        actual_batch_size = example.pop("batch_size")
        example.pop('scene_msg')
        for b in range(actual_batch_size):
            example_b = index_dict_of_batched_tensors_tf(example, b)

            count += 1

            def _custom_viz_t(scenario: ScenarioWithVisualization, e: Dict, t: int):
                if t > 0:
                    accept_probability_t = predictions['probabilities'][b, t - 1, 0].numpy()
                else:
                    accept_probability_t = -999
                scenario.plot_accept_probability(accept_probability_t)
                scenario.plot_traj_idx_rviz(batch_idx * batch_size + b)

            anim = RvizAnimation(myobj=view.dataset_loader.get_scenario(),
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
    return count
