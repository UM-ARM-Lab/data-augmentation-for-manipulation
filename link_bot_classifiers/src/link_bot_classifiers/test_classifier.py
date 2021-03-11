#!/usr/bin/env python
import pathlib
from typing import List, Optional, Callable, Dict

import numpy as np
import tensorflow as tf

from link_bot_classifiers import classifier_utils
from link_bot_classifiers.nn_classifier import NNClassifierWrapper
from link_bot_data.visualization import init_viz_env
from link_bot_data.viz_for_model import viz_transition_for_model_t_batched
from link_bot_gazebo import GazeboServices
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from merrrt_visualization.rviz_animation_controller import RvizAnimation
from moonshine.moonshine_utils import repeat, numpify, sequence_of_dicts_to_dict_of_tensors, add_time_dim
from state_space_dynamics import dynamics_utils


def test_classifier(classifier_model_dir: pathlib.Path,
                    fwd_model_dir: List[pathlib.Path],
                    n_actions: int,
                    saved_state: Optional[pathlib.Path],
                    generate_actions: Callable):
    fwd_model, _ = dynamics_utils.load_generic_model([pathlib.Path(p) for p in fwd_model_dir])
    classifier: NNClassifierWrapper = classifier_utils.load_generic_model([classifier_model_dir])

    service_provider = GazeboServices()
    service_provider.setup_env(verbose=0,
                               real_time_rate=0,
                               max_step_size=fwd_model.data_collection_params['max_step_size'],
                               play=False)
    if saved_state:
        service_provider.restore_from_bag(saved_state)

    scenario = fwd_model.scenario
    scenario.on_before_get_state_or_execute_action()

    # NOTE: perhaps it would make sense to have a "fwd_model" have API for get_env, get_state, sample_action, etc
    #  because the fwd_model knows it's scenario, and importantly it also knows it's data_collection_params
    #  which is what we're using here to pass to the scenario methods
    params = fwd_model.data_collection_params
    environment = numpify(scenario.get_environment(params))
    start_state = numpify(scenario.get_state())

    start_state_tiled = repeat(start_state, n_actions, axis=0, new_axis=True)
    start_states_tiled = add_time_dim(start_state_tiled)

    actions = generate_actions(environment, start_state_tiled, scenario, params, n_actions)

    environment_tiled = repeat(environment, n_actions, axis=0, new_axis=True)
    actions_dict = sequence_of_dicts_to_dict_of_tensors(actions)
    actions_dict = add_time_dim(actions_dict)
    predictions, _ = fwd_model.propagate_tf_batched(environment=environment_tiled,
                                                    state=start_states_tiled,
                                                    actions=actions_dict)

    # Run classifier
    state_sequence_length = 2
    accept_probabilities, _ = classifier.check_constraint_tf_batched(environment=environment_tiled,
                                                                     states=predictions,
                                                                     actions=actions_dict,
                                                                     state_sequence_length=state_sequence_length,
                                                                     batch_size=n_actions)
    # animate over the sampled actions
    anim = RvizAnimation(scenario=scenario,
                         n_time_steps=n_actions,
                         init_funcs=[init_viz_env],
                         t_funcs=[
                             lambda s, e, t: init_viz_env(s, e),
                             viz_transition_for_model_t_batched({}, fwd_model),
                             ExperimentScenario.plot_accept_probability_t,
                         ],
                         )
    example = {
        'accept_probability': tf.squeeze(accept_probabilities, axis=1),
    }
    example.update(environment)
    example.update(predictions)
    example.update(actions_dict)
    anim.play(example)


def sample_random_actions(environment: Dict,
                          start_state_tiled: Dict,
                          scenario: ExperimentScenario,
                          params: Dict,
                          n_actions: int):
    aciton_rng = np.random.RandomState(0)
    actions = scenario.sample_action_batch(environment=environment,
                                           state=start_state_tiled,
                                           action_params=params,
                                           action_rng=aciton_rng,
                                           validate=False,
                                           batch_size=n_actions)
    return actions
