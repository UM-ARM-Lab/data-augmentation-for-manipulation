#!/usr/bin/env python
import argparse
import json
import pathlib
import threading
from time import sleep
from typing import Dict

import colorama
import numpy as np

import rospy
from link_bot_planning.my_planner import PlanningQuery, PlanningResult
from link_bot_planning.plan_and_execute import ExecutionResult, TrialStatus
from link_bot_planning.results_utils import labeling_params_from_planner_params
from link_bot_pycommon.args import my_formatter, int_range_arg, int_set_arg
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.serialization import load_gzipped_pickle
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.moonshine_utils import numpify


def main():
    colorama.init(autoreset=True)
    np.set_printoptions(linewidth=250, precision=3, suppress=True)
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("results_dir", type=pathlib.Path, help='directory containing metrics.json')
    parser.add_argument("trial_idx", type=int_set_arg, help='which plan(s) to show')
    parser.add_argument("threshold", type=float)
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--filter-by-status", type=str, nargs="+")
    parser.add_argument("--show-tree", action="store_true")
    parser.add_argument("--verbose", '-v', action="count", default=0)

    rospy.init_node("plot_results")

    args = parser.parse_args()

    with (args.results_dir / 'metadata.json').open('r') as metadata_file:
        metadata_str = metadata_file.read()
        metadata = json.loads(metadata_str)
    scenario = get_scenario(metadata['scenario'])

    for trial_idx in args.trial_idx:
        results_filename = args.results_dir / f'{trial_idx}_metrics.pkl.gz'
        datum = load_gzipped_pickle(results_filename)

        if if_filter_with_status(datum, args.filter_by_status):
            print(f"Trial {trial_idx} ...")
            plot_steps(args.show_tree, scenario, datum, metadata, {'threshold': args.threshold}, args.verbose)
            print(f"... complete with status {datum['trial_status']}")


def get_goal_threshold(planner_params):
    if 'goal_params' in planner_params:
        goal_threshold = planner_params['goal_params']['threshold']
    else:
        goal_threshold = planner_params['goal_threshold']
    return goal_threshold


def if_filter_with_status(datum, filter_by_status):
    if not filter_by_status:
        return True

    # for step in datum['steps']:
    #     status = step['planning_result']['status']
    #     if status == 'MyPlannerStatus.NotProgressing':
    #         return True
    if datum['trial_status'] == TrialStatus.Timeout:
        return True

    return False


def plot_steps(show_tree: bool,
               scenario: ExperimentScenario,
               datum: Dict,
               metadata: Dict,
               fallback_labeing_params: Dict,
               verbose: int):
    planner_params = metadata['planner_params']
    goal_threshold = get_goal_threshold(planner_params)

    labeling_params = labeling_params_from_planner_params(planner_params, fallback_labeing_params)

    steps = datum['steps']

    if len(steps) == 0:
        q: PlanningQuery = datum['planning_queries'][0]
        start = q.start
        goal = q.goal
        environment = q.environment
        anim = RvizAnimationController(n_time_steps=1)
        scenario.plot_state_rviz(start, label='actual', color='#ff0000aa')
        scenario.plot_goal_rviz(goal, goal_threshold)
        scenario.plot_environment_rviz(environment)
        anim.step()
        return

    goal = datum['goal']
    first_step = steps[0]
    planning_query: PlanningQuery = first_step['planning_query']
    environment = numpify(planning_query.environment)
    all_actual_states = []
    types = []
    all_predicted_states = []
    all_actions = []

    actual_states = None
    predicted_states = None
    if len(steps) == 0:
        raise ValueError("no steps!")

    for step_idx, step in enumerate(steps):
        if verbose >= 1:
            print(step['type'])
        if step['type'] == 'executed_plan':
            planning_result: PlanningResult = step['planning_result']
            execution_result: ExecutionResult = step['execution_result']
            actions = planning_result.actions
            actual_states = execution_result.path
            predicted_states = planning_result.path
        elif step['type'] == 'executed_recovery':
            execution_result: ExecutionResult = step['execution_result']
            actions = [step['recovery_action']]
            actual_states = execution_result.path
            predicted_states = [None, None]
        else:
            raise NotImplementedError(f"invalid step type {step['type']}")

        actions = numpify(actions)
        actual_states = numpify(actual_states)
        predicted_states = numpify(predicted_states)

        all_actions.extend(actions)
        types.extend([step['type']] * len(actions))
        all_actual_states.extend(actual_states[:-1])
        all_predicted_states.extend(predicted_states[:-1])

        if show_tree and step['type'] == 'executed_plan':
            def _draw_tree_function(scenario, tree_json):
                print(f"n vertices {len(tree_json['vertices'])}")
                for vertex in tree_json['vertices']:
                    scenario.plot_tree_state(vertex, color='#77777722')
                    sleep(0.001)

            planning_result: PlanningResult = step['planning_result']
            tree_thread = threading.Thread(target=_draw_tree_function,
                                           args=(scenario, planning_result.tree,))
            tree_thread.start()

    # but do add the actual final states
    all_actual_states.append(actual_states[-1])
    all_predicted_states.append(predicted_states[-1])

    anim = RvizAnimationController(n_time_steps=len(all_actual_states))

    def _type_action_color(type_t: str):
        if type_t == 'executed_plan':
            return 'b'
        elif type_t == 'executed_recovery':
            return '#ff00ff'

    scenario.reset_planning_viz()
    while not anim.done:
        scenario.plot_environment_rviz(environment)
        t = anim.t()
        s_t = all_actual_states[t]
        s_t_pred = all_predicted_states[t]
        scenario.plot_state_rviz(s_t, label='actual', color='#ff0000aa')
        c = '#0000ffaa'
        if len(all_actions) > 0:
            if t < anim.max_t:
                type_t = types[t]
                action_color = _type_action_color(type_t)
                scenario.plot_action_rviz(s_t, all_actions[t], color=action_color)
            else:
                type_t = types[t - 1]
                action_color = _type_action_color(type_t)
                scenario.plot_action_rviz(all_actual_states[t - 1], all_actions[t - 1], color=action_color)

        if s_t_pred is not None:
            scenario.plot_state_rviz(s_t_pred, label='predicted', color=c)
            is_close = scenario.compute_label(s_t, s_t_pred, labeling_params)
            scenario.plot_is_close(is_close)
        else:
            scenario.plot_is_close(None)
        dist_to_goal = scenario.distance_to_goal(s_t, goal)
        actually_at_goal = dist_to_goal < goal_threshold
        scenario.plot_goal_rviz(goal, goal_threshold, actually_at_goal)
        anim.step()


if __name__ == '__main__':
    main()
