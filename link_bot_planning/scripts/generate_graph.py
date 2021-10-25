#!/usr/bin/env python

import argparse
import logging
import pathlib
import warnings
from time import time, perf_counter
from typing import Dict, List

import tensorflow as tf
from tqdm import tqdm

from arc_utilities import ros_init
from arc_utilities.algorithms import nested_dict_update
from link_bot_data.dataset_utils import add_predicted
from link_bot_gazebo.gazebo_services import GazeboServices
from link_bot_gazebo.gazebo_utils import get_gazebo_processes
from link_bot_planning.execute_full_tree import execute_full_tree
from link_bot_planning.get_planner import get_planner
from link_bot_planning.my_planner import PlanningQuery, PlanningResult, LoggingTree, are_states_close
from link_bot_planning.planning_evaluation import load_planner_params
from link_bot_planning.test_scenes import get_all_scenes, TestScene
from link_bot_planning.timeout_or_not_progressing import NExtensions
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from link_bot_pycommon.serialization import dump_gzipped_pickle, load_gzipped_pickle

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    from ompl import util as ou


def _are_states_close(a: Dict, b: Dict):
    keys = [add_predicted(k) for k in ['rope', 'left_gripper', 'right_gripper']]
    return are_states_close(a, b, keys)


def generate_execution_graph(gazebo_processes: List,
                             service_provider: GazeboServices,
                             scenario: ScenarioWithVisualization,
                             planning_query: PlanningQuery,
                             planning_result: PlanningResult,
                             planner_params: Dict,
                             verbose: int):
    [p.resume() for p in gazebo_processes]

    execution_gen = execute_full_tree(scenario=scenario,
                                      service_provider=service_provider,
                                      tree=planning_result.tree,
                                      planner_params=planner_params,
                                      planning_query=planning_query,
                                      verbose=verbose,
                                      stop_on_error=False)

    graph = LoggingTree()
    for e in tqdm(execution_gen, total=planning_result.tree.size):
        planned_before_state = {add_predicted(k): v for k, v in e.planned_before_state.items()}
        planned_after_state = {add_predicted(k): v for k, v in e.planned_after_state.items()}
        combined_before_state = nested_dict_update(planned_before_state, e.before_state)
        combined_after_state = nested_dict_update(planned_after_state, e.after_state)
        graph.add(before_state=combined_before_state,
                  action=e.action,
                  after_state=combined_after_state,
                  accept_probabilities=None,
                  are_states_close_f=_are_states_close)

    return graph


def generate_planning_graph(planning_query: PlanningQuery,
                            gazebo_processes: List,
                            goal: Dict,
                            planner_params: Dict,
                            scenario: ScenarioWithVisualization,
                            max_n_extensions: int,
                            verbose: int):
    goal['goal_type'] = planner_params['goal_params']['goal_type']
    rrt_planner = get_planner(planner_params, verbose=verbose, log_full_tree=True, scenario=scenario)

    def _override_ptc(_: PlanningQuery):
        return NExtensions(max_n_extensions=max_n_extensions)

    rrt_planner.make_ptc = _override_ptc

    [p.suspend() for p in gazebo_processes]
    planning_result = rrt_planner.plan(planning_query)
    return planning_query, planning_result


def generate_graph(root: pathlib.Path,
                   name: str,
                   planner_params: Dict,
                   scenario: ScenarioWithVisualization,
                   start: Dict,
                   goal: Dict,
                   verbose: int,
                   gazebo_processes: List,
                   max_n_extensions: int,
                   service_provider: GazeboServices,
                   restore_from_planning_tree: bool):
    planning_result_filename = root / f"{name}-planning_result.pkl.gz"
    planning_query = get_planning_query(goal, planner_params, scenario, start)

    if not restore_from_planning_tree:
        planning_result = generate_planning_graph(planning_query=planning_query,
                                                  gazebo_processes=gazebo_processes,
                                                  goal=goal,
                                                  planner_params=planner_params,
                                                  scenario=scenario,
                                                  verbose=verbose,
                                                  max_n_extensions=max_n_extensions)
        dump_gzipped_pickle(planning_result, planning_result_filename)
    else:
        print("Restoring!")
        with planning_result_filename.open("rb") as planning_result_file:
            planning_result = load_gzipped_pickle(planning_result_file)

    graph = generate_execution_graph(gazebo_processes=gazebo_processes,
                                     service_provider=service_provider,
                                     planning_query=planning_query,
                                     scenario=scenario,
                                     planning_result=planning_result,
                                     planner_params=planner_params,
                                     verbose=verbose)

    return planning_query, planning_result, graph, planning_query.environment


def get_planning_query(goal, planner_params, scenario, start):
    environment = scenario.get_environment(planner_params)
    planning_query = PlanningQuery(goal=goal,
                                   environment=environment,
                                   start=start,
                                   seed=0,
                                   trial_start_time_seconds=perf_counter())
    return planning_query


def generate_graph_data(name: str,
                        n_extensions,
                        planner_params_filename: pathlib.Path,
                        test_scene: TestScene,
                        verbose: int,
                        restore_from_planning_tree: bool):
    root = pathlib.Path("/media/shared/graphs")
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

    planning_query, planning_result, graph, env = generate_graph(root=root,
                                                                 name=name,
                                                                 planner_params=planner_params,
                                                                 scenario=scenario,
                                                                 start=start,
                                                                 goal=goal,
                                                                 verbose=verbose,
                                                                 gazebo_processes=gazebo_processes,
                                                                 max_n_extensions=n_extensions,
                                                                 service_provider=service_provider,
                                                                 restore_from_planning_tree=restore_from_planning_tree)

    out_filename = root / f"{name}.pkl.gz"
    graph_data = {
        'generated_at':    int(time()),
        'planning_query':  planning_query,
        'planning_result': planning_result,
        'env':             env,
        'graph':           graph,
        'start':           start,
        'goal':            goal,
        'params':          planner_params,
    }
    dump_gzipped_pickle(graph_data, out_filename)
    print(out_filename)


@ros_init.with_ros("generate_graph")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("planner_params", type=pathlib.Path)
    parser.add_argument("test_scenes", type=pathlib.Path)
    parser.add_argument("--n", '-n', type=int, default=10000)
    parser.add_argument("--restore-from-planning-tree", '-r', action='store_true')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")

    args = parser.parse_args()

    test_scenes = get_all_scenes(args.test_scenes)

    tf.get_logger().setLevel(logging.ERROR)
    ou.setLogLevel(ou.LOG_ERROR)

    for test_scene in test_scenes:
        generate_graph_data(name=args.name,
                            planner_params_filename=args.planner_params,
                            test_scene=test_scene,
                            verbose=args.verbose,
                            n_extensions=args.n,
                            restore_from_planning_tree=args.restore_from_planning_tree)


if __name__ == '__main__':
    main()
