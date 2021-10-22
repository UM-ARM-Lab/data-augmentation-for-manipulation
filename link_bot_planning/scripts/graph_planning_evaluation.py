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
from link_bot_planning.tree_utils import make_predicted_reached_goal, StateActionTree, trim_tree
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.serialization import load_gzipped_pickle
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(None)


def tree_eval_classifier(environment, tree, classifiers):
    def _eval(parent: StateActionTree):
        tree.classifier_probabilities = {}
        for child in parent.children:
            for c in classifiers:
                p_accepts_for_model = c.check_constraint(environment=environment,
                                                         states_sequence=[parent.state, child.state],
                                                         actions=[child.action])
                tree.classifier_probabilites[c.name] = p_accepts_for_model

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

    scenario = get_scenario(params['scenario'])

    t0 = perf_counter()

    classifiers = [load_generic_model(d, scenario=scenario) for d in classifier_model_dirs]

    trimmed_tree = StateActionTree()
    goal_cond = make_predicted_reached_goal(scenario, goal, goal_threshold)
    trim_tree(tree, trimmed_tree, goal_cond)
    trimmed_tree = trimmed_tree.children[0]

    tree_eval_classifier(environment, trimmed_tree, classifiers)

    planning_time = perf_counter() - t0

    results = {
        # 'paths':         paths,
        'planning_time': planning_time,
    }

    with (outdir / 'results.pkl').open("wb") as f:
        pickle.dump(results, f)


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
