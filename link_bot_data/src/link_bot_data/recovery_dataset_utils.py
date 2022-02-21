import pathlib
from copy import deepcopy
from time import perf_counter
from typing import Optional, List, Dict

import hjson
import numpy as np
import tensorflow as tf
from colorama import Fore

import rospy
from link_bot_classifiers import classifier_utils
from link_bot_classifiers.nn_classifier_wrapper import NNClassifierWrapper
from link_bot_data.dataset_utils import add_predicted
from link_bot_data.tf_dataset_utils import tf_write_example, deserialize_scene_msg, count_up_to_next_record_idx
from link_bot_data.dynamics_dataset import DynamicsDatasetLoader
from link_bot_data.recovery_dataset import compute_recovery_probabilities
from link_bot_data.visualization import init_viz_env, recovery_transition_viz_t, init_viz_action
from link_bot_pycommon.debugging_utils import debug_viz_batch_indices
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.grid_utils import batch_extent_to_origin_point_tf
from link_bot_pycommon.matplotlib_utils import adjust_lightness
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from link_bot_pycommon.serialization import my_hdump
from merrrt_visualization.rviz_animation_controller import RvizAnimation, RvizSimpleStepper
from moonshine.indexing import index_dict_of_batched_tensors_tf, index_batched_dict, index_batch_time
from moonshine.torch_and_tf_utils import repeat
from moonshine.tensorflow_utils import sequence_of_dicts_to_dict_of_tensors
from state_space_dynamics import dynamics_utils
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction

DEBUG = False


def make_recovery_dataset(dataset_dir: pathlib.Path,
                          fwd_model_dir,
                          classifier_model_dir: pathlib.Path,
                          labeling_params: pathlib.Path,
                          outdir: pathlib.Path,
                          batch_size: int,
                          use_gt_rope: bool,
                          start_at: Optional = None,
                          stop_at: Optional = None):
    labeling_params = hjson.load(labeling_params.open("r"))

    make_recovery_dataset_from_params_dict(dataset_dir=dataset_dir,
                                           fwd_model_dir=fwd_model_dir,
                                           classifier_model_dir=classifier_model_dir,
                                           labeling_params=labeling_params,
                                           outdir=outdir,
                                           batch_size=batch_size,
                                           use_gt_rope=use_gt_rope,
                                           start_at=start_at,
                                           stop_at=stop_at)


def make_recovery_dataset_from_params_dict(dataset_dir: pathlib.Path,
                                           fwd_model_dir,
                                           classifier_model_dir: pathlib.Path,
                                           labeling_params: Dict,
                                           outdir: pathlib.Path,
                                           batch_size: int,
                                           use_gt_rope: bool,
                                           start_at: Optional = None,
                                           stop_at: Optional = None):
    # append "best_checkpoint" before loading
    classifier_model_dir = classifier_model_dir / 'best_checkpoint'
    fwd_model_dir = pathlib.Path(fwd_model_dir) / 'best_checkpoint'

    np.random.seed(0)
    tf.random.set_seed(0)

    rospy.logwarn("hard coded scenario name here...")
    scenario = get_scenario("dual_arm_rope_sim_val_with_robot_feasibility_checking")
    classifier_model = classifier_utils.load_generic_model(classifier_model_dir, scenario)

    dynamics_hparams = hjson.load((dataset_dir / 'hparams.hjson').open('r'))
    fwd_model = dynamics_utils.load_generic_model(fwd_model_dir, scenario)

    dataset = DynamicsDatasetLoader([dataset_dir], use_gt_rope=use_gt_rope)

    outdir.mkdir(exist_ok=True)
    print(Fore.GREEN + f"Making recovery dataset {outdir.as_posix()}")
    new_hparams_filename = outdir / 'hparams.hjson'
    recovery_dataset_hparams = dynamics_hparams

    recovery_dataset_hparams['dataset_dir'] = dataset_dir
    recovery_dataset_hparams['fwd_model_dir'] = fwd_model_dir
    recovery_dataset_hparams['classifier_model'] = classifier_model_dir
    recovery_dataset_hparams['fwd_model_hparams'] = fwd_model.hparams
    recovery_dataset_hparams['labeling_params'] = labeling_params
    recovery_dataset_hparams['state_keys'] = fwd_model.state_keys
    recovery_dataset_hparams['action_keys'] = fwd_model.action_keys
    recovery_dataset_hparams['state_metadata_keys'] = dataset.state_metadata_keys
    recovery_dataset_hparams['env_keys'] = dataset.env_keys
    recovery_dataset_hparams['start_at'] = start_at
    recovery_dataset_hparams['stop_at'] = stop_at
    recovery_dataset_hparams['has_sampled_actions'] = True
    my_hdump(recovery_dataset_hparams, new_hparams_filename.open("w"), indent=2)

    outdir.mkdir(parents=True, exist_ok=True)

    start_at = progress_point(start_at)
    stop_at = progress_point(stop_at)

    modes = ['train', 'val', 'test']
    for mode in modes:
        if start_at is not None and modes.index(mode) < modes.index(start_at[0]):
            continue
        if stop_at is not None and modes.index(mode) > modes.index(stop_at[0]):
            continue

        tf_dataset_for_mode = dataset.get_datasets(mode=mode)

        full_output_directory = outdir / mode
        full_output_directory.mkdir(parents=True, exist_ok=True)

        # figure out that record_idx to start at
        record_idx = count_up_to_next_record_idx(full_output_directory)

        # FIXME: start_at is not implemented correctly in the sense that it shouldn't be the same
        #  across train/val/test
        for out_example in generate_recovery_examples(tf_dataset=tf_dataset_for_mode,
                                                      modes=modes,
                                                      mode=mode,
                                                      fwd_model=fwd_model,
                                                      classifier_model=classifier_model,
                                                      dataset=dataset,
                                                      labeling_params=labeling_params,
                                                      batch_size=batch_size,
                                                      start_at=start_at,
                                                      stop_at=stop_at):
            for batch_idx in range(out_example['traj_idx'].shape[0]):
                out_example_b = index_dict_of_batched_tensors_tf(out_example, batch_idx)

                # if DEBUG:
                #     visualize_recovery_generation(scenario, dataset, fwd_model, out_example_b, labeling_params)
                #
                tf_write_example(full_output_directory, out_example_b, record_idx)
                record_idx += 1

    return outdir


def visualize_recovery_generation(scenario, dataset, fwd_model, out_example_b, labeling_params):
    viz_out_example_b = deepcopy(out_example_b)
    recovery_probability = compute_recovery_probabilities(viz_out_example_b['accept_probabilities'],
                                                          labeling_params['n_action_samples'])
    viz_out_example_b['recovery_probability'] = recovery_probability
    anim = RvizAnimation(scenario=scenario,
                         n_time_steps=labeling_params['action_sequence_horizon'],
                         init_funcs=[init_viz_env,
                                     init_viz_action(dataset.scenario_metadata, fwd_model.action_keys,
                                                     fwd_model.state_keys),
                                     ],
                         t_funcs=[init_viz_env,
                                  recovery_transition_viz_t(dataset.scenario_metadata,
                                                            fwd_model.state_keys + dataset.state_metadata_keys),
                                  lambda s, e, t: scenario.plot_recovery_probability_t(e, t),
                                  ])
    deserialize_scene_msg(viz_out_example_b)
    anim.play(viz_out_example_b)


def progress_point(start_at):
    if start_at is not None:
        start_at_mode, start_at_batch_idx = start_at.split(":")
        start_at = (start_at_mode, int(start_at_batch_idx))
    return start_at


def generate_recovery_examples(tf_dataset: tf.data.Dataset,
                               modes: List[str],
                               mode: str,
                               fwd_model,
                               classifier_model,
                               dataset,
                               labeling_params,
                               batch_size,
                               start_at,
                               stop_at):
    action_sequence_horizon = labeling_params['action_sequence_horizon']
    tf_dataset = tf_dataset.batch(batch_size)
    action_rng = np.random.RandomState(0)
    n_batches = 0
    for _ in tf_dataset:
        n_batches += 1

    t0 = perf_counter()
    for in_batch_idx, example in enumerate(tf_dataset):
        if start_at is not None and (modes.index(mode) == modes.index(start_at[0]) and in_batch_idx < start_at[1]):
            continue
        if stop_at is not None and (modes.index(mode) == modes.index(stop_at[0]) and in_batch_idx >= stop_at[1]):
            print(Fore.GREEN + "Done!" + Fore.RESET)
            return

        dt = perf_counter() - t0
        print(Fore.GREEN + f"{mode}: {in_batch_idx}/{n_batches}, {dt:.3f}s" + Fore.RESET)
        actual_batch_size = int(example['traj_idx'].shape[0])

        # iterate over every subsequence of exactly length actions_sequence_horizon
        for start_t in range(0, dataset.steps_per_traj - action_sequence_horizon + 1, labeling_params['start_step']):
            end_t = start_t + action_sequence_horizon

            dataset.state_metadata_keys = ['joint_names']  # NOTE: perhaps ACOs should be state metadata?
            state_keys = fwd_model.state_keys + dataset.state_metadata_keys
            actual_states_from_start_t = {k: example[k][:, start_t:end_t] for k in state_keys}
            actions_from_start_t = {k: example[k][:, start_t:end_t - 1] for k in fwd_model.action_keys}

            data = (example,
                    actions_from_start_t,
                    actual_states_from_start_t,
                    labeling_params,
                    dataset.data_collection_params,
                    )
            constants = (actual_batch_size,
                         action_sequence_horizon,
                         classifier_model.horizon,
                         start_t,
                         end_t,
                         dataset.env_keys,
                         )
            out_examples = generate_recovery_actions_examples(fwd_model=fwd_model,
                                                              classifier_model=classifier_model,
                                                              data=data,
                                                              constants=constants,
                                                              action_rng=action_rng)
            yield out_examples


def generate_recovery_actions_examples(fwd_model: BaseDynamicsFunction,
                                       classifier_model: NNClassifierWrapper,
                                       data,
                                       constants,
                                       action_rng: np.random.RandomState):
    example, actual_actions, actual_states, labeling_params, data_collection_params = data
    batch_size, action_sequence_horizon, classifier_horizon, start_t, end_t, env_keys = constants
    scenario = fwd_model.scenario

    environment = {k: example[k] for k in env_keys}
    if 'origin_point' not in environment:
        environment['origin_point'] = batch_extent_to_origin_point_tf(example['extent'], example['res'])

    all_accept_probabilities = []
    all_actions_dicts = []
    for ast in range(action_sequence_horizon):  # ast = action sequence time. Just using "t" was confusing
        # Sample actions
        n_action_samples = labeling_params['n_action_samples']
        n_actions = classifier_horizon - 1
        actual_state_t = index_dict_of_batched_tensors_tf(actual_states, ast, batch_axis=1)
        state_tiled_reps = [1, n_action_samples, 1, 1]
        actual_states_tiled = {k: tf.tile(v[:, tf.newaxis], state_tiled_reps) for k, v in actual_states.items()}
        # [t:t+1] to keep dim, as opposed to just [t]
        start_states_tiled_t = index_dict_of_batched_tensors_tf(actual_states_tiled, ast, batch_axis=2, keep_dims=True)

        # elements in random_actions_dict should have shape [batch_size, n_action_samples, n_actions, n]
        random_actions_dict = batch_stateless_sample_action(scenario=scenario,
                                                            environment=environment,
                                                            state=actual_state_t,
                                                            batch_size=batch_size,
                                                            n_action_samples=n_action_samples,
                                                            n_actions=n_actions,
                                                            action_params=data_collection_params,
                                                            action_rng=action_rng)
        all_actions_dicts.append(random_actions_dict)

        # NOTE: perhaps write generic "collapse" functions to merging (unmerging?) dimensions?
        bs = batch_size * n_action_samples
        # TODO: can we use the fact that all examples in the batch are the same environment?
        environment_tiled_batched = repeat(environment, n_action_samples, 0, False)
        start_states_tiled_t_batched = {k: tf.reshape(v, [bs, -1]) for k, v in start_states_tiled_t.items()}
        random_actions_dict_batched = {k: tf.reshape(v, [bs, 1, -1]) for k, v in random_actions_dict.items()}

        predictions, accept_probabilities = predict_and_classify_for_recovery_dataset(
            fwd_model,
            classifier_model,
            environment_tiled_batched,  # [b*nas, ...]
            start_states_tiled_t_batched,  # [b*nas, 1, ...]
            random_actions_dict_batched,  # [b*nas, 1, ...]
            bs,
            classifier_horizon)

        # reshape to separate batch from sampled actions
        accept_probabilities = tf.reshape(accept_probabilities, [batch_size, n_action_samples])
        predictions = {k: tf.reshape(v, [batch_size, n_action_samples, classifier_horizon, -1]) for k, v in
                       predictions.items()}

        all_accept_probabilities.append(accept_probabilities)

        if DEBUG:
            viz_generate_example_batch(accept_probabilities, start_states_tiled_t, classifier_model, environment,
                                       n_action_samples, predictions, random_actions_dict, scenario, batch_size)

    # NOTE: just store all examples with their probabilities, we can filter later, which is more flexible
    #  so we want avoid doing that many times
    all_accept_probabilities = tf.stack(all_accept_probabilities, axis=1)

    # TODO: include predictions and the sampled actions. Including these is not easy,
    #  because the right way to do this would be to have nested structure, but that's not supported by TF datasets API
    out_examples = {
        'traj_idx':             example['traj_idx'],
        'start_t':              tf.stack([start_t] * batch_size),
        'end_t':                tf.stack([end_t] * batch_size),
        'accept_probabilities': all_accept_probabilities,
    }
    out_examples.update(environment)
    # add true start states
    out_examples.update(actual_states)
    out_examples.update(actual_actions)
    all_sampled_actions_dict = sequence_of_dicts_to_dict_of_tensors(all_actions_dicts, axis=1)
    all_sampled_actions_dict = {"sampled_actions_" + k: v for k, v in all_sampled_actions_dict.items()}
    out_examples.update(all_sampled_actions_dict)

    return out_examples


def predict_and_classify_for_recovery_dataset(fwd_model, classifier_model, environment, actual_states,
                                              random_actions_dict, batch_size, state_sequence_length):
    # Predict
    mean_dynamics_predictions, _ = fwd_model.propagate_tf_batched(environment=environment,
                                                                  start_state=actual_states,
                                                                  actions=random_actions_dict)

    # Check classifier
    accept_probabilities = classifier_model.check_constraint_tf_batched(environment=environment,
                                                                        states=mean_dynamics_predictions,
                                                                        actions=random_actions_dict,
                                                                        batch_size=batch_size,
                                                                        state_sequence_length=state_sequence_length)

    return mean_dynamics_predictions, accept_probabilities


def viz_generate_example_batch(accept_probabilities, start_states_tiled_t, classifier_model, environment,
                               n_action_samples,
                               predictions, random_actions_dict, scenario, batch_size):
    stepper = RvizSimpleStepper()
    for b in debug_viz_batch_indices(batch_size):
        recovery_probabilities = compute_recovery_probabilities(accept_probabilities, n_action_samples)
        environment_b = index_batched_dict(environment, b)
        deserialize_scene_msg(environment_b)
        start_state_b = index_batched_dict(start_states_tiled_t, b)
        predictions_b = index_batched_dict(predictions, b)
        actions_b = index_batched_dict(random_actions_dict, b)
        accept_probabilities_b = accept_probabilities[b]
        recovery_probability = recovery_probabilities[b]
        viz_example_b = {}
        viz_example_b.update(environment_b)
        viz_example_b.update(start_state_b)
        viz_example_b.update({add_predicted(k): v for k, v in predictions_b.items()})
        viz_example_b.update(actions_b)
        viz_example_b['accept_probabilities'] = accept_probabilities_b.numpy()

        viz_generate_example_single(classifier_model, n_action_samples, scenario, viz_example_b, recovery_probability)
        stepper.step()


def viz_generate_example_single(classifier_model, n_action_samples, scenario, viz_example_b, recovery_probability):
    def _init_viz_state(scenario, example):
        c_actual = '#ff0000'
        before_state = index_batch_time(example, classifier_model.state_keys, b=0, t=0)
        scenario.plot_state_rviz(before_state, label='before', color=c_actual)

    def _viz_i(scenario: ScenarioWithVisualization, example: Dict, i: int):
        action = {k: example[k][i, 0] for k in classifier_model.action_keys}
        state_for_action = index_batch_time(example, classifier_model.state_keys, b=i, t=0)
        scenario.plot_action_rviz(state_for_action, action)

        scenario.plot_recovery_probability(recovery_probability)

        accept_probability_i = example['accept_probabilities'][i]
        scenario.plot_accept_probability(accept_probability_i)

        c_pred = '#0000ff'
        pred_state_keys = [add_predicted(k) for k in classifier_model.state_keys]
        before_state = index_batch_time(example, pred_state_keys, b=i, t=0)
        scenario.plot_state_rviz(before_state, label='before_pred', color=c_pred)
        after_state = index_batch_time(example, pred_state_keys, b=i, t=1)
        scenario.plot_state_rviz(after_state, label='after_pred', color=adjust_lightness(c_pred, 0.5))

    anim = RvizAnimation(scenario=scenario,
                         n_time_steps=n_action_samples,
                         init_funcs=[init_viz_env, _init_viz_state],
                         t_funcs=[init_viz_env, _viz_i])
    anim.play(viz_example_b)


def batch_stateless_sample_action(scenario: ScenarioWithVisualization,
                                  environment: Dict,
                                  state: Dict,
                                  batch_size: int,
                                  n_action_samples: int,
                                  n_actions: int,
                                  action_params: Dict,
                                  action_rng: np.random.RandomState):
    # TODO: make the lowest level sample_action operate on batched state dictionaries so this is less annoying
    all_actions = []
    for b in range(batch_size):
        state_b = index_batched_dict(state, b)
        environment_b = index_batched_dict(environment, b)
        deserialize_scene_msg(environment_b)
        action_sequences_b = scenario.sample_action_sequences(environment=environment_b,
                                                              state=state_b,
                                                              action_params=action_params,
                                                              n_action_sequences=n_action_samples,
                                                              action_sequence_length=n_actions,
                                                              validate=True,
                                                              # validate=False,
                                                              action_rng=action_rng)
        action_sequences_b = [sequence_of_dicts_to_dict_of_tensors(a) for a in action_sequences_b]
        action_sequences_b = sequence_of_dicts_to_dict_of_tensors(action_sequences_b)
        all_actions.append(action_sequences_b)
    all_actions = sequence_of_dicts_to_dict_of_tensors(all_actions)
    return all_actions


def recovering_mask(needs_recovery):
    """
    Looks for the first occurrence of the pattern [1, 0] in each row (appending a 0 to each row first),
    and the index where this is found defines the end of the mask.
    The first time step is masked to False because it will always be 1
    :param needs_recovery: float matrix [B,H] but all values should be 0.0 or 1.0
    :return: boolean matrix [B,H]
    """
    batch_size, horizon = needs_recovery.shape

    range = tf.stack([tf.range(horizon)] * batch_size, axis=0)
    mask = tf.cumsum(needs_recovery, axis=1) > range
    has_a_0 = tf.reduce_any(needs_recovery == 0, axis=1, keepdims=True)
    starts_with_1 = needs_recovery[:, 0:1] == 1
    trues = tf.cast(tf.ones([batch_size, 1]), tf.bool)
    mask_final = tf.concat([trues, mask[:, :-1]], axis=1)
    mask_final = tf.logical_and(tf.logical_and(mask_final, has_a_0), starts_with_1)
    return mask_final
