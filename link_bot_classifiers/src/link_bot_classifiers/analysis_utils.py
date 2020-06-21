from typing import Dict, List

import numpy as np
import tensorflow as tf

from link_bot_planning.plan_and_execute import execute_actions
from link_bot_pycommon.base_services import BaseServices
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.pycommon import make_dict_tf_float32
from link_bot_pycommon.ros_pycommon import make_movable_object_services, \
    get_environment_for_extents_3d
from moonshine.moonshine_utils import numpify, sequence_of_dicts_to_dict_of_tensors
from state_space_dynamics.model_utils import EnsembleDynamicsFunction


def predict(fwd_models: EnsembleDynamicsFunction,
            environment: Dict,
            start_states: List[Dict],
            actions: List[List[List[Dict]]],
            n_actions: int,
            n_actions_sampled: int,
            n_start_states: int) -> List[List[List[Dict]]]:
    # reformat the inputs to be efficiently batched
    actions_dict = {}
    for actions_for_start_state in actions:
        for actions in actions_for_start_state:
            for action in actions:
                for k, v in action.items():
                    if k not in actions_dict:
                        actions_dict[k] = []
                    actions_dict[k].append(v)

    actions_batched = {k: tf.reshape(v, [n_actions_sampled * n_start_states, n_actions, -1]) for k, v in actions_dict.items()}
    start_states = sequence_of_dicts_to_dict_of_tensors(start_states)
    start_states = make_dict_tf_float32({k: tf.expand_dims(v, axis=1) for k, v in start_states.items()})
    # copy from start states for each random action
    start_states_tiled = {k: tf.concat([v] * n_actions_sampled, axis=0) for k, v in start_states.items()}

    # Actually do the predictions
    predictions_dict = fwd_models.propagate_differentiable_batched(start_states=start_states_tiled, actions=actions_batched)

    # break out the num actions and num start states
    n_states = n_actions + 1
    predictions_dict = {k: tf.reshape(v, [n_start_states, n_actions_sampled, n_states, -1]) for k, v in predictions_dict.items()}

    # invert structure to List[List[List[Dict]]] and return
    predictions_list = []
    for i in range(n_start_states):
        predictions_i = []
        for j in range(n_actions_sampled):
            predictions_ij = []
            for t in range(n_actions + 1):
                prediction_ijt = {k: predictions_dict[k][i, j, t] for k, v in predictions_dict.items()}
                predictions_ij.append(prediction_ijt)
            predictions_i.append(predictions_ij)
        predictions_list.append(predictions_i)
    return predictions_list


def execute(service_provider: BaseServices,
            scenario: ExperimentScenario,
            start_states: List[Dict],
            random_actions: List[List[List[Dict]]],
            ):
    actual_state_sequences = []
    for start_state, actions_for_start_state in zip(start_states, random_actions):
        actual_state_sequences_for_start_state = []
        for actions in actions_for_start_state:
            # reset to the start state
            scenario.teleport_to_state(numpify(start_state))
            # execute actions and record the observed states
            actual_states = execute_actions(service_provider, scenario, start_state, actions)
            actual_state_sequences_for_start_state.append(actual_states)
        # reset when done for conveniently re-running the script
        actual_state_sequences.append(actual_state_sequences_for_start_state)
        scenario.teleport_to_state(numpify(start_state))
    return actual_state_sequences


def setup(service_provider: BaseServices, fwd_model: EnsembleDynamicsFunction, test_params: Dict, real_time_rate: float = 0):
    max_step_size = fwd_model.data_collection_params['max_step_size']
    service_provider.setup_env(verbose=0, real_time_rate=real_time_rate, max_step_size=max_step_size)

    movable_object_services = {k: make_movable_object_services(k) for k in test_params['object_positions'].keys()}
    fwd_model.scenario.move_objects_to_positions(movable_object_services, test_params['object_positions'])

    environment = get_environment_for_extents_3d(extent=test_params['extent'],
                                                 res=fwd_model.data_collection_params['res'],
                                                 service_provider=service_provider,
                                                 robot_name=fwd_model.scenario.robot_name())
    return environment


def predict_and_execute(service_provider,
                        fwd_model: EnsembleDynamicsFunction,
                        environment: Dict,
                        start_states: List[Dict],
                        actions: List[List[List[Dict]]],
                        n_actions: int,
                        n_actions_sampled: int,
                        n_start_states: int):
    # Prediction
    predictions = predict(fwd_models=fwd_model,
                          environment=environment,
                          start_states=start_states,
                          actions=actions,
                          n_actions=n_actions,
                          n_actions_sampled=n_actions_sampled,
                          n_start_states=n_start_states)
    # Execute
    scenario = fwd_model.scenario
    actual_states_lists = execute(service_provider, scenario, start_states, actions)

    return fwd_model, environment, actual_states_lists, predictions


def sample_actions(scenario: ExperimentScenario, environment: Dict, start_states: List[Dict], n_samples: int, horizon: int):
    action_rng = np.random.RandomState(0)
    action_sequences = []
    for i, start_state in enumerate(start_states):
        action_sequences_for_start_state = []
        for j in range(n_samples):
            action_sequence = []
            for t in range(horizon):
                action = scenario.sample_action(environment=environment,
                                                state=start_state,
                                                params={},
                                                action_rng=action_rng)
                action_sequence.append(action)
            action_sequences_for_start_state.append(action_sequence)
        action_sequences.append(action_sequences_for_start_state)
    return action_sequences
