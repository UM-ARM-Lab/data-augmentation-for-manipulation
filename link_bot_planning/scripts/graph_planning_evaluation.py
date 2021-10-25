#!/usr/bin/env python
import argparse
import logging
import pathlib
import pickle
from time import perf_counter
from typing import List

import colorama
import tensorflow as tf

from arc_utilities import ros_init
from link_bot_classifiers.classifier_utils import load_generic_model
from link_bot_planning.my_planner import LoggingTree, PlanningResult, MyPlannerStatus
from link_bot_planning.tree_utils import make_predicted_reached_goal, trim_tree, tree_to_paths
from link_bot_planning.trial_result import ExecutionResult
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.serialization import load_gzipped_pickle
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(None)


def tree_eval_classifier(environment, tree, classifiers):
    def _eval(parent: LoggingTree):
        for child in parent.children:
            child.accept_probabilities = {}
            for c in classifiers:
                p_accepts_for_model = c.check_constraint(environment=environment,
                                                         states_sequence=[parent.state, child.state],
                                                         actions=[child.action])
                child.accept_probabilities[c.name] = p_accepts_for_model

            _eval(child)

    _eval(tree)


def graph_planning_evaluation(outdir: pathlib.Path,
                              classifier_model_dirs: List[pathlib.Path],
                              graph_filename: pathlib.Path,
                              goal_threshold: float,
                              verbose: int = 0):
    graph_data = load_gzipped_pickle(graph_filename)
    tree = graph_data['graph']  # it's actually a tree
    environment = graph_data['env']
    start = graph_data['start']
    goal = graph_data['goal']
    params = graph_data['params']
    planning_query = graph_data['planning_query']
    full_planning_result = graph_data['planning_result']

    scenario = get_scenario(params['scenario'])

    t0 = perf_counter()

    classifiers = [load_generic_model(d, scenario=scenario) for d in classifier_model_dirs]

    trimmed_tree = LoggingTree()
    goal_cond = make_predicted_reached_goal(scenario, goal, goal_threshold)
    trim_tree(tree, trimmed_tree, goal_cond)
    trimmed_tree = trimmed_tree.children[0]

    tree_eval_classifier(environment, trimmed_tree, classifiers)

    planning_time = perf_counter() - t0
    paths = list(tree_to_paths(trimmed_tree))

    outdir.mkdir(exist_ok=True)

    for i, path in enumerate(paths):
        path_states = [p_i['state'] for p_i in path]
        path_actions = [p_i['action'] for p_i in path]
        step_planning_result = PlanningResult(
            path=path_states,
            actions=path_actions,
            status=MyPlannerStatus.Solved,
            tree=LoggingTree(),
            time=0.0,
            mean_propagate_time=0.0
        )
        steps = [{
            'type':             'executed_plan',
            'planning_query':   planning_query,
            'planning_result':  step_planning_result,
            'recovery_action':  None,
            'execution_result': ExecutionResult(
                path=path_states,
                end_trial=False,
                stopped=False,
                end_t=-1,
            ),
            'time_since_start': None,
        }]

        trial_datum = {
            'graph_filename':        graph_filename,
            'classifier_model_dirs': classifier_model_dirs,
            'scenario':              params['scenario'],
            'planning_time':         planning_time,
            'paths':                 paths,
            'goal':                  goal,
            'steps':                 steps,
            'end_state':             path_states[-1],
            'total_time':            -1,
            'metadata':              {
                'planner_params': params
            }
        }
        classifier_model_dirs_str = [c.as_posix() for c in classifier_model_dirs]
        trial_datum['metadata']['planner_params']['classifier_model_dir'] = classifier_model_dirs_str
        trial_datum['metadata']['planner_params']['goal_params']['threshold'] = goal_threshold

        outfilename = outdir / f'trial_{i}.pkl'
        with outfilename.open("wb") as f:
            pickle.dump(trial_datum, f)
        print(f"{outfilename.as_posix()}")


@ros_init.with_ros("graph_planning_evaluation")
def main():
    colorama.init(autoreset=True)
    tf.get_logger().setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument("graph", type=pathlib.Path)
    parser.add_argument('classifier', type=pathlib.Path)
    parser.add_argument("outdir", type=pathlib.Path, help='used in making the output directory')
    parser.add_argument('--goal-threshold', type=float, default=0.045)
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")

    args = parser.parse_args()

    classifier_model_dirs = [args.classifier, pathlib.Path("cl_trials/new_feasibility_baseline/none")]

    graph_planning_evaluation(outdir=args.outdir,
                              classifier_model_dirs=classifier_model_dirs,
                              graph_filename=args.graph,
                              goal_threshold=args.goal_threshold,
                              verbose=args.verbose)


if __name__ == '__main__':
    main()
