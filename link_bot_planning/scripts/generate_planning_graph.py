import argparse
import logging
import pathlib
import warnings
from time import time, perf_counter
from typing import Dict

import tensorflow as tf

from arc_utilities import ros_init
from link_bot_gazebo.gazebo_services import GazeboServices
from link_bot_gazebo.gazebo_utils import get_gazebo_processes
from link_bot_planning.get_planner import get_planner
from link_bot_planning.my_planner import PlanningQuery
from link_bot_planning.planning_evaluation import load_planner_params
from link_bot_planning.test_scenes import get_all_scenes, TestScene
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.serialization import dump_gzipped_pickle

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    from ompl import util as ou


def generate_planning_graph(planner_params: Dict, scenario, start, goal, verbose: int, gazebo_processes):
    goal['goal_type'] = planner_params['goal_params']['goal_type']
    rrt_planner = get_planner(planner_params, verbose=verbose, log_full_tree=True, scenario=scenario)
    environment = scenario.get_environment(planner_params)
    planning_query = PlanningQuery(goal=goal,
                                   environment=environment,
                                   start=start,
                                   seed=0,
                                   trial_start_time_seconds=perf_counter())

    [p.suspend() for p in gazebo_processes]
    planning_result = rrt_planner.plan(planning_query)

    return planning_result.tree


def generate_planning_graph_data(name: str,
                                 planner_params_filename: pathlib.Path,
                                 test_scene: TestScene,
                                 verbose: int):
    root = pathlib.Path("/media/shared/planning_graphs")
    root.mkdir(exist_ok=True)

    planner_params = load_planner_params(planner_params_filename)

    gazebo_processes = get_gazebo_processes()
    [p.resume() for p in gazebo_processes]

    scenario = get_scenario(planner_params['scenario'])
    scenario.on_before_get_state_or_execute_action()
    service_provider = GazeboServices()
    scenario.restore_from_bag(service_provider, planner_params, test_scene.get_scene_filename())

    start = scenario.get_state()
    goal = test_scene.goal

    graph = generate_planning_graph(planner_params=planner_params,
                                    scenario=scenario,
                                    start=start,
                                    goal=goal,
                                    verbose=verbose,
                                    gazebo_processes=gazebo_processes)

    outfilename = root / f"{name}.pkl.gz"
    graph_data = {
        'generated-at': int(time()),
        'graph':        graph,
        'start':        start,
        'goal':         goal,
        'params':       planner_params,
    }
    dump_gzipped_pickle(graph_data, outfilename)


@ros_init.with_ros("generate_planning_graph")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("planner_params", type=pathlib.Path)
    parser.add_argument("test_scenes", type=pathlib.Path)
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")

    args = parser.parse_args()

    test_scenes = get_all_scenes(args.test_scenes)

    tf.get_logger().setLevel(logging.ERROR)
    ou.setLogLevel(ou.LOG_ERROR)

    for test_scene in test_scenes:
        generate_planning_graph_data(name=args.name,
                                     planner_params_filename=args.planner_params,
                                     test_scene=test_scene,
                                     verbose=args.verbose)


if __name__ == '__main__':
    main()
