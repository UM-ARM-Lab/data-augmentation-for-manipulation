import pathlib
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Optional

import hjson
import tensorflow as tf
from progressbar import progressbar

import rospy
from link_bot_data.classifier_dataset import ClassifierDatasetLoader
from link_bot_data.dataset_utils import add_predicted, add_label, deserialize_scene_msg, write_example
from link_bot_data.load_dataset import load_dynamics_dataset
from link_bot_data.progressbar_widgets import mywidgets
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.serialization import my_hdump
from moonshine.filepath_tools import load_hjson
from moonshine.indexing import index_dict_of_batched_tensors_tf
from moonshine.moonshine_utils import gather_dict, numpify
from state_space_dynamics import dynamics_utils
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


@dataclass
class PredictionActualExample:
    example: Dict
    actual_states: Dict
    actions: Dict
    predictions: Dict
    env_keys: List[str]
    start_t: int
    labeling_params: Dict
    actual_prediction_horizon: int
    batch_size: int


def make_classifier_dataset(dataset_dir: pathlib.Path,
                            fwd_model_dir: pathlib.Path,
                            labeling_params: pathlib.Path,
                            outdir: pathlib.Path,
                            use_gt_rope: bool,
                            visualize: bool,
                            save_format: str,
                            batch_size,
                            start_at: Optional[int] = None,
                            stop_at: Optional[int] = None,
                            custom_threshold: Optional[float] = None,
                            ):
    labeling_params = load_hjson(labeling_params)
    make_classifier_dataset_from_params_dict(dataset_dir=dataset_dir,
                                             fwd_model_dir=fwd_model_dir,
                                             labeling_params=labeling_params,
                                             outdir=outdir,
                                             use_gt_rope=use_gt_rope,
                                             visualize=visualize,
                                             save_format=save_format,
                                             custom_threshold=custom_threshold,
                                             take=None,
                                             batch_size=batch_size,
                                             start_at=start_at,
                                             stop_at=stop_at,
                                             )


def make_classifier_dataset_from_params_dict(dataset_dir: pathlib.Path,
                                             fwd_model_dir: pathlib.Path,
                                             labeling_params: Dict,
                                             outdir: pathlib.Path,
                                             use_gt_rope: bool,
                                             visualize: bool,
                                             save_format: str,
                                             custom_threshold: Optional[float] = None,
                                             take: Optional[int] = None,
                                             batch_size: Optional[int] = None,
                                             start_at: Optional[int] = None,
                                             stop_at: Optional[int] = None):
    if labeling_params.get('includes_starts_far'):
        rospy.logwarn('including examples where the start actual vs predicted are far')

    # append "best_checkpoint" before loading
    fwd_model_dir = fwd_model_dir / 'best_checkpoint'

    dynamics_hparams = hjson.load((dataset_dir / 'hparams.hjson').open('r'))

    dataset_loader = load_dynamics_dataset([dataset_dir])

    fwd_models = dynamics_utils.load_generic_model(fwd_model_dir, dataset_loader.scenario)

    new_hparams_filename = outdir / 'hparams.hjson'
    classifier_dataset_hparams = dynamics_hparams

    classifier_dataset_hparams['dataset_dir'] = dataset_dir.as_posix()
    classifier_dataset_hparams['fwd_model_hparams'] = fwd_models.hparams
    classifier_dataset_hparams['labeling_params'] = labeling_params
    classifier_dataset_hparams['env_keys'] = dataset_loader.env_keys
    classifier_dataset_hparams['true_state_keys'] = dataset_loader.state_keys
    classifier_dataset_hparams['state_metadata_keys'] = dataset_loader.state_metadata_keys
    classifier_dataset_hparams['predicted_state_keys'] = fwd_models.state_keys
    classifier_dataset_hparams['action_keys'] = dataset_loader.action_keys
    classifier_dataset_hparams['start-at'] = start_at
    classifier_dataset_hparams['stop-at'] = stop_at
    my_hdump(classifier_dataset_hparams, new_hparams_filename.open("w"), indent=2)

    # because we're currently making this dataset, we can't call "get_dataset" but we can still use it to visualize
    # a bit hacky...
    classifier_dataset_for_viz = ClassifierDatasetLoader([outdir], use_gt_rope=use_gt_rope)

    if custom_threshold is not None:
        labeling_params['threshold'] = custom_threshold

    t0 = perf_counter()
    total_example_idx = 0
    for mode in ['train', 'val', 'test']:
        dataset = dataset_loader.get_datasets(mode=mode, take=take)

        if save_format == 'tfrecords':
            full_output_directory = outdir / mode
            full_output_directory.mkdir(parents=True, exist_ok=True)
        elif save_format == 'pkl':
            full_output_directory = outdir
        else:
            raise NotImplementedError()

        out_examples_gen = generate_classifier_examples(fwd_models, dataset, dataset_loader, labeling_params,
                                                        batch_size)
        for out_examples in out_examples_gen:
            for out_examples_start_t in out_examples:
                actual_batch_size = out_examples_start_t['traj_idx'].shape[0]
                for batch_idx in range(actual_batch_size):
                    out_example_b = index_dict_of_batched_tensors_tf(out_examples_start_t, batch_idx)
                    out_example_b['metadata'] = {k: v[batch_idx] for k, v in out_examples_start_t['metadata'].items()}

                    if out_example_b['time_idx'].ndim == 0:
                        continue

                    if visualize:
                        add_label(out_example_b, labeling_params['threshold'])
                        classifier_dataset_for_viz.anim_transition_rviz(out_example_b)

                    write_example(full_output_directory, out_example_b, total_example_idx, save_format)
                    # rospy.loginfo_throttle(10, f"Examples: {total_example_idx:10d}, Time: {perf_counter() - t0:.3f}")
                    total_example_idx += 1

    return outdir


def generate_classifier_examples(fwd_model: BaseDynamicsFunction,
                                 dataset,
                                 dataset_loader,
                                 labeling_params: Dict,
                                 batch_size: int):
    classifier_horizon = labeling_params['classifier_horizon']
    assert classifier_horizon >= 2
    # dataset = batch_tf_dataset(dataset, batch_size, drop_remainder=False)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
    sc = dataset_loader.get_scenario()

    for idx, _ in enumerate(dataset):
        pass
    n_total_batches = idx

    t0 = perf_counter()
    for idx, example in enumerate(progressbar(dataset, widgts=mywidgets)):
        deserialize_scene_msg(example)

        dt = perf_counter() - t0
        actual_batch_size = int(example['traj_idx'].shape[0])

        valid_out_examples = []
        for start_t in range(0, dataset_loader.steps_per_traj - classifier_horizon + 1, labeling_params['start_step']):
            prediction_end_t = dataset_loader.steps_per_traj
            actual_prediction_horizon = prediction_end_t - start_t
            dataset_loader.state_metadata_keys = ['joint_names']  # NOTE: perhaps ACOs should be state metadata?
            state_keys = dataset_loader.state_keys + dataset_loader.state_metadata_keys
            actual_states_from_start_t = {k: example[k][:, start_t:prediction_end_t] for k in state_keys}
            actual_start_states = {k: example[k][:, start_t] for k in state_keys}
            actions_from_start_t = {k: example[k][:, start_t:prediction_end_t - 1] for k in dataset_loader.action_keys}
            environment = {k: example[k] for k in dataset_loader.env_keys}

            predictions_from_start_t, _ = fwd_model.propagate_tf_batched(environment=environment,
                                                                         start_state=actual_start_states,
                                                                         actions=actions_from_start_t)
            prediction_actual = PredictionActualExample(example=example,
                                                        actions=actions_from_start_t,
                                                        actual_states=actual_states_from_start_t,
                                                        predictions=predictions_from_start_t,
                                                        start_t=start_t,
                                                        env_keys=dataset_loader.env_keys,
                                                        labeling_params=labeling_params,
                                                        actual_prediction_horizon=actual_prediction_horizon,
                                                        batch_size=actual_batch_size)
            valid_out_examples_for_start_t = generate_classifier_examples_from_batch(sc, prediction_actual)
            valid_out_examples.extend(valid_out_examples_for_start_t)

        yield valid_out_examples


def generate_classifier_examples_from_batch(scenario: ExperimentScenario, prediction_actual: PredictionActualExample):
    labeling_params = prediction_actual.labeling_params
    prediction_horizon = prediction_actual.actual_prediction_horizon
    classifier_horizon = labeling_params['classifier_horizon']
    batch_size = prediction_actual.batch_size

    valid_out_example_batches = []
    for classifier_start_t in range(0, prediction_horizon - classifier_horizon + 1):
        classifier_end_t = classifier_start_t + classifier_horizon

        prediction_start_t = prediction_actual.start_t
        prediction_start_t_batched = int_scalar_to_batched_float(batch_size, prediction_start_t)
        classifier_start_t_batched = int_scalar_to_batched_float(batch_size, classifier_start_t)
        classifier_end_t_batched = int_scalar_to_batched_float(batch_size, classifier_end_t)
        out_example = {
            'traj_idx':           prediction_actual.example['traj_idx'],
            'prediction_start_t': prediction_start_t_batched,
            'classifier_start_t': classifier_start_t_batched,
            'classifier_end_t':   classifier_end_t_batched,
        }
        out_example.update({k: prediction_actual.example[k] for k in prediction_actual.env_keys})

        # this slice gives arrays of fixed length (ex, 5) which must be null padded from out_example_end_idx onwards
        sliced_actual, sliced_predictions = slice_to_fixed_length(classifier_horizon,
                                                                  classifier_start_t,
                                                                  out_example,
                                                                  prediction_actual)

        add_perception_reliability(scenario=scenario,
                                   actual=sliced_actual,
                                   predictions=sliced_predictions,
                                   labeling_params=labeling_params,
                                   out_example=out_example)

        # compute label
        valid_out_examples = add_model_error_and_filter(scenario, sliced_actual, sliced_predictions, out_example,
                                                        labeling_params,
                                                        prediction_actual.batch_size)
        valid_out_examples_np = numpify(valid_out_examples)
        valid_out_examples_np['metadata'] = {
            'error': out_example['error'],
        }
        valid_out_example_batches.append(valid_out_examples_np)

    return valid_out_example_batches


def add_model_error_and_filter(scenario, actual, predictions, out_example, labeling_params: Dict, batch_size: int):
    is_close = add_model_error(actual, labeling_params, out_example, predictions, scenario)

    valid_out_examples = filter_valid_example_batches(is_close,
                                                      labeling_params,
                                                      out_example,
                                                      batch_size)
    return valid_out_examples


def add_model_error(actual, labeling_params, out_example, predictions, scenario):
    threshold = labeling_params['threshold']
    error = scenario.classifier_distance(actual, predictions)
    out_example['error'] = tf.cast(error, dtype=tf.float32)
    is_close = error < threshold
    return is_close


def slice_to_fixed_length(classifier_horizon: int,
                          classifier_start_t: int,
                          out_example: Dict,
                          prediction_actual: PredictionActualExample):
    state_slice = slice(classifier_start_t, classifier_start_t + classifier_horizon)
    action_slice = slice(classifier_start_t, classifier_start_t + classifier_horizon - 1)
    sliced_actual = {}
    for key, actual_state_component in prediction_actual.actual_states.items():
        actual_state_component_sliced = actual_state_component[:, state_slice]
        out_example[key] = actual_state_component_sliced
        sliced_actual[key] = actual_state_component_sliced
    sliced_predictions = {}
    for key, prediction_component in prediction_actual.predictions.items():
        prediction_component_sliced = prediction_component[:, state_slice]
        out_example[add_predicted(key)] = prediction_component_sliced
        sliced_predictions[key] = prediction_component_sliced
    sliced_actions = {}
    for key, action_component in prediction_actual.actions.items():
        action_component_sliced = action_component[:, action_slice]
        out_example[key] = action_component_sliced
        sliced_actions[key] = action_component_sliced
    return sliced_actual, sliced_predictions


def int_scalar_to_batched_float(batch_size: int, t: int):
    return tf.cast(tf.stack([t] * batch_size, axis=0), tf.float32)


def filter_valid_example_batches(is_close, labeling_params: Dict, examples: Dict, batch_size: int):
    if not labeling_params.get('includes_starts_far', False):
        is_first_predicted_state_close = is_close[:, 0]
        valid_indices = tf.where(is_first_predicted_state_close)
        valid_indices = tf.squeeze(valid_indices, axis=1)
    else:
        valid_indices = tf.range(batch_size, dtype=tf.int64)
    # keep only valid_indices from every key in out_example...
    valid_out_examples = gather_dict(examples, valid_indices)
    return valid_out_examples


def add_perception_reliability(scenario: ExperimentScenario,
                               actual: Dict,
                               predictions: Dict,
                               labeling_params: Dict,
                               out_example: Dict):
    if 'perception_reliability_method' in labeling_params:
        pr_method = labeling_params['perception_reliability_method']
        if pr_method == 'gt':
            perception_reliability = gt_perception_reliability(scenario, actual, predictions)
            out_example['perception_reliability'] = perception_reliability
        else:
            raise NotImplementedError(f"unrecognized perception reliability method {pr_method}")


def zero_through_inf_to_one_through_zero(x):
    """ maps [0, inf) to [1, 0) """
    return 1 / (1 + x)


def gt_perception_reliability(scenario: ExperimentScenario, actual: Dict, predicted: Dict):
    gt_perception_error_bt = scenario.classifier_distance(actual, predicted)
    # add over time
    gt_perception_error_b = tf.math.reduce_sum(gt_perception_error_bt, axis=1)
    perception_reliability = zero_through_inf_to_one_through_zero(gt_perception_error_b)
    return perception_reliability


def batch_of_many_of_actions_sequences_to_dict(actions, n_actions_sampled, n_start_states, n_actions):
    # reformat the inputs to be efficiently batched
    actions_dict = {}
    for actions_for_start_state in actions:
        for actions in actions_for_start_state:
            for action in actions:
                for k, v in action.items():
                    if k not in actions_dict:
                        actions_dict[k] = []
                    actions_dict[k].append(v)
    actions_batched = {k: tf.reshape(v, [n_actions_sampled * n_start_states, n_actions, -1])
                       for k, v in actions_dict.items()}
    return actions_batched
