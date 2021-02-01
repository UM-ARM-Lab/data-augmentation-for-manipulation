#!/usr/bin/env python
import argparse
import json
import pathlib

import colorama
import numpy as np

import rospy
from link_bot_classifiers import classifier_utils
from link_bot_classifiers.nn_classifier import NNClassifierWrapper
from link_bot_data.visualization import init_viz_env, viz_state_action_for_model_t
from link_bot_planning.results_utils import get_paths
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.pycommon import paths_from_json
from link_bot_pycommon.serialization import load_gzipped_pickle
from merrrt_visualization.rviz_animation_controller import RvizAnimation
from moonshine.moonshine_utils import sequence_of_dicts_to_dict_of_sequences
from state_space_dynamics import dynamics_utils


def main():
    colorama.init(autoreset=True)
    np.set_printoptions(precision=3, suppress=True, linewidth=200)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("results_dir", type=pathlib.Path)
    parser.add_argument("trial_idx", type=int)
    parser.add_argument("time_step_idx", type=int)

    args = parser.parse_args()

    rospy.init_node('test_classifier_on_saved_results')

    metadata_filename = args.results_dir / 'metadata.json'
    with metadata_filename.open('r') as metadata_file:
        metadata = json.load(metadata_file)
    planner_params = metadata['planner_params']
    fwd_model_dirs = paths_from_json(planner_params['fwd_model_dir'])
    classifier_model_dirs = paths_from_json(planner_params['classifier_model_dir'])

    n_actions = 1

    fwd_model, _ = dynamics_utils.load_generic_model(fwd_model_dirs)
    classifier: NNClassifierWrapper = classifier_utils.load_generic_model(classifier_model_dirs)

    scenario = get_scenario(planner_params['scenario'])

    results_filename = args.results_dir / f'{args.trial_idx}_metrics.pkl.gz'
    datum = load_gzipped_pickle(results_filename)
    all_actions, all_actual_states, all_predicted_states, _ = get_paths(datum, scenario)
    action = all_actions[args.time_step_idx]
    s0_actual = all_actual_states[args.time_step_idx]
    s0_pred = all_predicted_states[args.time_step_idx]
    s1_actual = all_actual_states[args.time_step_idx + 1]
    s1_pred = all_predicted_states[args.time_step_idx + 1]

    environment = datum['planning_queries'][0].environment
    actions = [action]

    # Run classifier
    accept_probabilities, _ = classifier.check_constraint(environment=environment,
                                                          states_sequence=[s0_pred, s1_pred],
                                                          actions=actions)
    # animate
    predictions_dict = sequence_of_dicts_to_dict_of_sequences([s0_pred, s1_pred])
    actual_dict = sequence_of_dicts_to_dict_of_sequences([s0_actual, s1_actual])
    actions_dict = sequence_of_dicts_to_dict_of_sequences(actions)

    anim = RvizAnimation(scenario=scenario,
                         n_time_steps=(n_actions + 1),
                         init_funcs=[init_viz_env],
                         t_funcs=[
                             lambda s, e, t: init_viz_env(s, e),
                             viz_state_action_for_model_t({}, fwd_model),
                             ExperimentScenario.plot_accept_probability_t,  # need a free func here, not a bound one
                         ],
                         )
    example = {
        'accept_probability': accept_probabilities,
    }
    example.update(environment)
    example.update(predictions_dict)
    example.update(actual_dict)
    example.update(actions_dict)
    anim.play(example)


if __name__ == '__main__':
    main()
