#!/usr/bin/env python
import argparse
import gzip
import json
import pathlib
from time import sleep

import colorama
import numpy as np
from colorama import Fore

import rospy
from link_bot_classifiers import classifier_utils
from moonshine.gpu_config import limit_gpu_mem
from moonshine.tensorflow_utils import listify
from moonshine.numpify import numpify
from state_space_dynamics import dynamics_utils

limit_gpu_mem(7)


def main():
    colorama.init(autoreset=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=pathlib.Path, help='dir containing *_metrics.json.gz')
    parser.add_argument("plan_idx", type=int)

    args = parser.parse_args()

    rospy.init_node("postprocess_result")

    with (args.results_dir / 'metadata.json').open('r') as metadata_file:
        metadata_str = metadata_file.read()
        metadata = json.loads(metadata_str)

    planner_params = metadata['planner_params']
    fwd_model_dirs = pathlib.Path(planner_params['fwd_model_dir'])
    fwd_model = dynamics_utils.load_generic_model(fwd_model_dirs)
    scenario = fwd_model.scenario

    classifier_model_dir = pathlib.Path(planner_params['classifier_model_dir'])
    classifier = classifier_utils.load_generic_model(classifier_model_dir, scenario=scenario)

    metrics_filename = args.results_dir / f"{args.plan_idx}_metrics.json.gz"
    with gzip.open(metrics_filename, 'rb') as f:
        data_str = f.read()
    datum = json.loads(data_str.decode("utf-8"))
    steps = datum['steps']

    goal = datum['goal']
    first_step = steps[0]
    environment = numpify(first_step['planning_query']['environment'])
    all_output_actions = []
    for step in steps:
        if step['type'] == 'executed_plan':
            actions = postprocess_step(scenario, fwd_model, classifier, environment, step, goal, planner_params)
            all_output_actions.extend(actions)
        elif step['type'] == 'executed_recovery':
            actions = [step['recovery_action']]
            all_output_actions.extend(actions)

    # viz the whole thing
    start_state = steps[0]['planning_result']['path'][0]
    final_states = fwd_model.propagate(environment, start_state, actions)
    T = len(final_states)
    for t, s_t in enumerate(final_states):
        scenario.plot_state_rviz(s_t, idx=t, label='smoothed', color='#00ff0099')
        if t < T - 1:
            scenario.plot_action_rviz(s_t, actions[t], idx=t, color="#ffffff99", label='smoothed')
        sleep(0.02)

    # Save the output actions
    outfilename = args.results_dir / f"{args.plan_idx}_smoothed.json"
    with outfilename.open("w") as outfile:
        json.dump(listify(all_output_actions), outfile)

    print(Fore.GREEN + f"Wrote {outfilename}" + Fore.RESET)


def postprocess_step(scenario, fwd_model, classifier, environment, step, goal, planner_params, verbose: int):
    actions = step['planning_result']['actions']
    predicted_states = step['planning_result']['path']
    plan_length = len(predicted_states)

    # if verbose >= 2:
    #     # visualize the initial plan
    #     scenario.reset_planning_viz()
    #     scenario.plot_environment_rviz(environment)
    #     for t, s_t in enumerate(predicted_states):
    #         scenario.plot_state_rviz(s_t, idx=t, label='planned', color='#ff000033')
    #         if t < plan_length - 1:
    #             a_t = actions[t]
    #             scenario.plot_action_rviz(s_t, a_t, idx=t, color="#0000ff33")
    #         sleep(0.02)

    rng = np.random.RandomState(0)

    for j in range(200):

        # randomly sample a start index
        start_t = rng.randint(0, plan_length - 2)

        # sample an end index
        end_t = rng.randint(min(start_t, plan_length - 2), min(start_t + 10, plan_length - 1))

        # interpolate the grippers
        start_state = predicted_states[start_t]
        end_state = predicted_states[end_t]
        interpolated_actions = scenario.interpolate(start_state, end_state)

        if verbose >= 2:
            scenario.plot_state_rviz(start_state, idx=0, label='from', color='y')
            scenario.plot_state_rviz(end_state, idx=1, label='to', color='m')

        accept_shortcut = True
        interpolated_state = start_state
        interpolated_states = [start_state]
        for interpolated_action in interpolated_actions:
            if verbose >= 2:
                scenario.plot_state_rviz(interpolated_state, label='interpolated')
                scenario.plot_action_rviz(interpolated_state, interpolated_action, label='interpolated')

            # propagate and check the classifier
            states = fwd_model.propagate(environment, interpolated_state, [interpolated_action])
            assert len(states) == 2
            accept_probabilities = classifier.check_constraint(environment, states, [interpolated_action])
            assert len(accept_probabilities) == 1
            accept_probability = accept_probabilities[0]

            # if accepted, replace the actions
            if accept_probability < 0.5:
                accept_shortcut = False
                break

            interpolated_state = states[-1]
            interpolated_states.append(states[-1])
        if accept_shortcut:
            actions = actions[:start_t] + interpolated_actions + actions[end_t:]
            predicted_states = predicted_states[:start_t] + interpolated_states + predicted_states[end_t + 1:]

            if verbose >= 2:
                for t, s_t in enumerate(predicted_states):
                    scenario.plot_state_rviz(s_t, idx=t, label='smoothed', color='#00ff0099')
                    if t < len(predicted_states) - 1:
                        scenario.plot_action_rviz(s_t, actions[t], idx=t, color="#ffffff99", label='smoothed')
                    sleep(0.01)
            print("smoothed")

    # Plot the smoothed result
    if verbose >= 2:
        final_states = fwd_model.propagate(environment, predicted_states[0], actions)
        plan_length = len(predicted_states)
        for t, s_t in enumerate(final_states):
            scenario.plot_state_rviz(s_t, idx=t, label='smoothed', color='#00ff0099')
            if t < plan_length - 1:
                scenario.plot_action_rviz(s_t, actions[t], idx=t, color="#ffffff99", label='smoothed')
            sleep(0.02)

        final_final_state = final_states[-1]
        goal_threshold = planner_params['goal_params']['threshold']
        still_reaches_goal = scenario.distance_to_goal(goal, final_final_state) < goal_threshold + 1e-3
        print(f"Still near goal? {still_reaches_goal}")

    return actions


if __name__ == "__main__":
    main()
