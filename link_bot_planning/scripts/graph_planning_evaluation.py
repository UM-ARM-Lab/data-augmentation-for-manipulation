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
from link_bot_planning.a_star_solver import AStarSolver
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.serialization import load_gzipped_pickle
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(None)


def graph_planning_evaluation(outdir: pathlib.Path,
                              classifier_model_dirs: List[pathlib.Path],
                              graph_filename: pathlib.Path,
                              verbose: int = 0):
    graph_data = load_gzipped_pickle(graph_filename)
    graph = graph_data['graph']
    start = graph_data['start']
    goal = graph_data['goal']
    params = graph_data['params']

    scenario = get_scenario(params['scenario'])

    t0 = perf_counter()
    solution = AStarSolver(scenario, graph).astar(start, goal)
    planning_time = perf_counter() - t0

    results = {
        'soultion':      solution,
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
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")

    args = parser.parse_args()

    classifier_model_dirs = [args.classifier, pathlib.Path("cl_trials/new_feasibility_baseline/none")]

    graph_planning_evaluation(outdir=args.outdir,
                              classifier_model_dirs=classifier_model_dirs,
                              graph_filename=args.graph,
                              verbose=args.verbose)


if __name__ == '__main__':
    main()
