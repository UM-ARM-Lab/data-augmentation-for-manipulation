import pathlib
from time import perf_counter
from typing import Optional, List, Dict

import hjson
import numpy as np
import tensorflow as tf
from colorama import Fore

from link_bot_classifiers import classifier_utils
from link_bot_classifiers.nn_classifier import NNClassifierWrapper
from link_bot_data.dataset_utils import tf_write_example, count_up_to_next_record_idx
from link_bot_data.dynamics_dataset import DynamicsDatasetLoader
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.pycommon import make_dict_tf_float32
from link_bot_pycommon.serialization import my_hdump
from moonshine.moonshine_utils import sequence_of_dicts_to_dict_of_tensors
from moonshine.indexing import index_dict_of_batched_tensors_tf
from state_space_dynamics import dynamics_utils
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


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
    if not isinstance(fwd_model_dir, List):
        fwd_model_dir = [fwd_model_dir]
    fwd_model_dir = [p / 'best_checkpoint' for p in fwd_model_dir]

    np.random.seed(0)
    tf.random.set_seed(0)

    dynamics_hparams = hjson.load((dataset_dir / 'hparams.hjson').open('r'))
    fwd_model, _ = dynamics_utils.load_generic_model(fwd_model_dir)

    dataset = DynamicsDatasetLoader([dataset_dir], use_gt_rope=use_gt_rope)

    outdir.mkdir(exist_ok=True)
    print(Fore.GREEN + f"Making recovery dataset {outdir.as_posix()}")
    new_hparams_filename = outdir / 'hparams.hjson'
    recovery_dataset_hparams = dynamics_hparams

    scenario = fwd_model.scenario
    if not isinstance(classifier_model_dir, List):
        classifier_model_dir = [classifier_model_dir]
    classifier_model = classifier_utils.load_generic_model(classifier_model_dir, scenario)

    recovery_dataset_hparams['dataset_dir'] = dataset_dir
    recovery_dataset_hparams['fwd_model_dir'] = fwd_model_dir
    recovery_dataset_hparams['classifier_model'] = classifier_model_dir
    recovery_dataset_hparams['fwd_model_hparams'] = fwd_model.hparams
    recovery_dataset_hparams['labeling_params'] = labeling_params
    recovery_dataset_hparams['state_keys'] = fwd_model.state_keys
    recovery_dataset_hparams['action_keys'] = fwd_model.action_keys
    recovery_dataset_hparams['start-at'] = start_at
    recovery_dataset_hparams['stop-at'] = stop_at
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
            # FIXME: is there an extra time/batch dimension?
            for batch_idx in range(out_example['traj_idx'].shape[0]):
                out_example_b = index_dict_of_batched_tensors_tf(out_example, batch_idx)

                # # BEGIN DEBUG
                # from link_bot_data.visualization import init_viz_env, recovery_transition_viz_t, init_viz_action
                # from copy import deepcopy
                #
                # viz_out_example_b = deepcopy(out_example_b)
                # recovery_probability = compute_recovery_probabilities(viz_out_example_b['accept_probabilities'],
                #                                                       labeling_params['n_action_samples'])
                # viz_out_example_b['recovery_probability'] = recovery_probability
                # anim = RvizAnimation(scenario=scenario,
                #                      n_time_steps=labeling_params['action_sequence_horizon'],
                #                      init_funcs=[init_viz_env,
                #                                  init_viz_action(dataset.scenario_metadata, fwd_model.action_keys,
                #                                                  fwd_model.state_keys),
                #                                  ],
                #                      t_funcs=[init_viz_env,
                #                               recovery_transition_viz_t(dataset.scenario_metadata,
                #                                                         fwd_model.state_keys),
                #                               lambda s, e, t: scenario.plot_recovery_probability_t(e, t),
                #                               ])
                # anim.play(viz_out_example_b)
                # # END DEBUG

                tf_write_example(full_output_directory, out_example_b, record_idx)
                record_idx += 1

    return outdir


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

            actual_states_from_start_t = {k: example[k][:, start_t:end_t] for k in fwd_model.state_keys}
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
                         actual_batch_size,
                         start_t,
                         end_t)
            out_examples = generate_recovery_actions_examples(fwd_model=fwd_model,
                                                              classifier_model=classifier_model,
                                                              scenario_metadata=dataset.scenario_metadata,
                                                              data=data,
                                                              constants=constants,
                                                              action_rng=action_rng)
            yield out_examples


def batch_stateless_sample_action(scenario: ExperimentScenario,
                                  environment: Dict,
                                  state: Dict,
                                  batch_size: int,
                                  n_action_samples: int,
                                  n_actions: int,
                                  action_params: Dict,
                                  action_rng: np.random.RandomState):
    # TODO: make the lowest level sample_action operate on batched state dictionaries
    action_sequences = scenario.sample_action_sequences(environment=environment,
                                                        state=state,
                                                        action_params=action_params,
                                                        n_action_sequences=n_action_samples,
                                                        action_sequence_length=n_actions,
                                                        validate=False,
                                                        action_rng=action_rng)
    action_sequences = [sequence_of_dicts_to_dict_of_tensors(a) for a in action_sequences]
    action_sequences = sequence_of_dicts_to_dict_of_tensors(action_sequences)
    return {k: tf.tile(v[tf.newaxis], [batch_size, 1, 1, 1]) for k, v in action_sequences.items()}


def generate_recovery_actions_examples(fwd_model: BaseDynamicsFunction,
                                       classifier_model: NNClassifierWrapper,
                                       scenario_metadata: Dict,
                                       data,
                                       constants,
                                       action_rng: np.random.RandomState):
    example, actual_actions, actual_states, labeling_params, data_collection_params = data
    actual_batch_size, action_sequence_horizon, classifier_horizon, batch_size, start_t, end_t = constants
    scenario = fwd_model.scenario

    full_env = example['env']
    full_env_origin = example['origin']
    full_env_extent = example['extent']
    full_env_res = example['res']
    environment = {
        'env': full_env,
        'origin': full_env_origin,
        'extent': full_env_extent,
        'res': full_env_res,
    }

    all_accept_probabilities = []
    for ast in range(action_sequence_horizon):  # ast = action sequence time. Just using "t" was confusing
        # Sample actions
        n_action_samples = labeling_params['n_action_samples']
        n_actions = classifier_horizon - 1
        actual_state_t = index_dict_of_batched_tensors_tf(actual_states, ast, batch_axis=1)
        actual_states_tiled = {k: tf.tile(v[:, tf.newaxis], [1, n_action_samples, 1, 1]) for k, v in
                               actual_states.items()}
        # [t:t+1] to keep dim, as opposed to just [t]
        start_states_tiled_t = index_dict_of_batched_tensors_tf(actual_states_tiled, ast, batch_axis=2, keep_dims=True)
        bs = actual_batch_size * n_action_samples
        # TODO: write generic "collapse" functions to merging dimensions
        start_states_tiled_t_batched = {k: tf.reshape(v, [bs, 1, -1]) for k, v in start_states_tiled_t.items()}

        # TODO: check we're sampling action sequences correctly here
        #  I think it should be that for we sample a number of action sequences independently, but
        #  that across the batch dimension the actions can be the same.
        random_actions_dict = batch_stateless_sample_action(scenario=scenario,
                                                            environment=environment,
                                                            state=actual_state_t,
                                                            batch_size=actual_batch_size,
                                                            n_action_samples=n_action_samples,
                                                            n_actions=n_actions,
                                                            action_params=data_collection_params,
                                                            action_rng=action_rng)
        random_actions_dict_batched = {k: tf.reshape(v, [bs, 1, -1]) for k, v in random_actions_dict.items()}

        def _predict_and_classify(_actual_states, _random_actions_dict):
            # Predict
            mean_dynamics_predictions, _ = fwd_model.propagate_differentiable_batched(environment=environment,
                                                                                      state=_actual_states,
                                                                                      actions=_random_actions_dict)

            # Check classifier
            environment_tiled = {k: tf.concat([v] * n_action_samples, axis=0) for k, v in environment.items()}
            accept_probabilities, _ = classifier_model.check_constraint_batched_tf(environment=environment_tiled,
                                                                                   predictions=mean_dynamics_predictions,
                                                                                   actions=_random_actions_dict,
                                                                                   batch_size=bs,
                                                                                   state_sequence_length=classifier_horizon)

            return mean_dynamics_predictions, accept_probabilities

        predictions, accept_probabilities = _predict_and_classify(start_states_tiled_t_batched,
                                                                  random_actions_dict_batched)

        # reshape to separate batch from sampled actions
        accept_probabilities = tf.reshape(accept_probabilities, [batch_size, n_action_samples])
        predictions = {k: tf.reshape(v, [batch_size, n_action_samples, classifier_horizon, -1]) for k, v in
                       predictions.items()}

        all_accept_probabilities.append(accept_probabilities)

        # recovery_probabilities = compute_recovery_probabilities(accept_probabilities, n_action_samples)
        # print(ast, accept_probabilities[0])

        # # BEGIN DEBUG
        # from link_bot_data.dataset_utils import add_predicted, index_batch_time_with_metadata
        # from link_bot_data.recovery_dataset import compute_recovery_probabilities
        # from matplotlib import cm
        # environment_b = index_dict_of_batched_tensors_tf(environment, 0)
        # actual_states_b = index_dict_of_batched_tensors_tf(actual_states_tiled, 0)
        # predictions_b = index_dict_of_batched_tensors_tf(predictions, 0)
        # actions_b = index_dict_of_batched_tensors_tf(random_actions_dict, 0)
        # accept_probabilities_b = accept_probabilities[0]
        # recovery_probability = recovery_probabilities[0]
        # viz_example_b = {}
        # viz_example_b.update(environment_b)
        # viz_example_b.update(actual_states_b)
        # viz_example_b.update({add_predicted(k): v for k, v in predictions_b.items()})
        # viz_example_b.update(actions_b)
        # viz_example_b['accept_probabilities'] = accept_probabilities_b.numpy()
        # pred_state_keys = classifier_model.pred_state_keys
        #
        # def _init_viz_true_action(scenario, example):
        #     pred_0 = index_batch_time_with_metadata(scenario_metadata, example, fwd_model.state_keys, b=0, t=ast)
        #     action = {k: actual_actions[k][0, 0] for k in fwd_model.action_keys}
        #     scenario.plot_action_rviz(pred_0, action, label='true action')
        #
        # def _init_viz_start_state(scenario, example):
        #     start_state = index_batch_time_with_metadata(scenario_metadata, example, fwd_model.state_keys, b=0, t=ast)
        #     scenario.plot_state_rviz(start_state, label='pred', color='#ff3333aa')
        #
        # def _viz_action_i(scenario: ExperimentScenario, example: Dict, i: int):
        #     action = {k: example[k][i, 0] for k in fwd_model.action_keys}
        #     pred_t = index_batch_time_with_metadata(scenario_metadata, example, fwd_model.state_keys, b=i, t=ast)
        #     scenario.plot_action_rviz(pred_t, action, color=cm.Blues(accept_probabilities_b[i]))
        #
        # def _recovery_transition_viz_i(scenario: ExperimentScenario, example: Dict, i: int):
        #     e_t_next = index_batch_time_with_metadata(scenario_metadata, example, pred_state_keys, b=i, t=1)
        #     scenario.plot_state_rviz(e_t_next, label='pred', color='#ff3333aa')
        #
        # anim = RvizAnimation(scenario=scenario,
        #                      n_time_steps=n_action_samples,
        #                      init_funcs=[init_viz_env,
        #                                  lambda s, e: scenario.plot_recovery_probability(recovery_probability),
        #                                  _init_viz_start_state,
        #                                  _init_viz_true_action,
        #                                  ],
        #                      t_funcs=[init_viz_env,
        #                               _recovery_transition_viz_i,
        #                               _viz_action_i,
        #                               lambda s, e, i: scenario.plot_accept_probability(e['accept_probabilities'][i]),
        #                               ])
        #
        # anim.play(viz_example_b)
        # # END DEBUG

    # NOTE: just store all examples with their probabilities, we can filter later, which is more flexible
    #  so we want avoid doing that many times
    all_accept_probabilities = tf.stack(all_accept_probabilities, axis=1)

    # TODO: include predictions and the sampled actions. Including these is not easy,
    #  because the right way to do this would be to have nested structure, but that's not supported by TF datasets API
    out_examples = {
        'env': full_env,
        'origin': full_env_origin,
        'extent': full_env_extent,
        'res': full_env_res,
        'traj_idx': example['traj_idx'],
        'start_t': tf.stack([start_t] * batch_size),
        'end_t': tf.stack([end_t] * batch_size),
        'accept_probabilities': all_accept_probabilities,
    }
    # add true start states
    out_examples.update(actual_states)
    out_examples.update(actual_actions)
    out_examples = make_dict_tf_float32(out_examples)

    return out_examples


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
