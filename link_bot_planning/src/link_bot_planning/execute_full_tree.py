import dataclasses
import pathlib
import tempfile
from typing import Dict, List

from arm_robots.robot import RobotPlanningError
from gazebo_msgs.msg import LinkStates
from link_bot_planning.my_planner import PlanningQuery
from link_bot_planning.test_scenes import get_states_to_save, save_test_scene_given_name
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization


@dataclasses.dataclass
class FullTreeExecutionElement:
    environment: Dict
    action: Dict
    before_state: Dict
    planned_before_state: Dict
    after_state: Dict
    planned_after_state: Dict
    depth: int
    accept_probabilities: List


def execute_full_tree(scenario: ScenarioWithVisualization,
                      service_provider,
                      tree,
                      planner_params: Dict,
                      planning_query: PlanningQuery,
                      bagfile_name=None,
                      depth: int = 0,
                      verbose: int = 1,
                      stop_on_error: bool = True):
    if depth == 0:
        service_provider.play()
        scenario.on_before_get_state_or_execute_action()
        scenario.grasp_rope_endpoints(settling_time=0.0)

    if bagfile_name is None:
        bagfile_name = store_bagfile()

    for child in tree.children:
        # if we only have one child we can skip the restore, this speeds things up a lot
        must_restore = len(tree.children) > 1
        if must_restore:
            scenario.restore_from_bag_rushed(service_provider=service_provider,
                                             params=planner_params,
                                             bagfile_name=bagfile_name)
        before_state, after_state, error = execute(scenario=scenario,
                                                   service_provider=service_provider,
                                                   environment=planning_query.environment,
                                                   action=child.action)
        if error and stop_on_error:
            continue

        # only include this example and continue the DFS if we were able to successfully execute the action
        yield FullTreeExecutionElement(environment=planning_query.environment,
                                       action=child.action,
                                       before_state=before_state,
                                       planned_before_state=tree.state,
                                       after_state=after_state,
                                       planned_after_state=child.state,
                                       depth=depth,
                                       accept_probabilities=child.accept_probabilities)

        # recursion
        yield from execute_full_tree(scenario=scenario,
                                     service_provider=service_provider,
                                     tree=child,
                                     planner_params=planner_params,
                                     planning_query=planning_query,
                                     bagfile_name=None,
                                     depth=depth + 1,
                                     verbose=verbose)


def execute(scenario, service_provider, environment: Dict, action: Dict):
    service_provider.play()
    before_state = scenario.get_state()
    error = False
    try:
        scenario.execute_action(environment=environment, state=before_state, action=action)
    except (RobotPlanningError, RuntimeError):
        error = True
    after_state = scenario.get_state()
    return before_state, after_state, error


def store_bagfile():
    joint_state, links_states = get_states_to_save()
    make_links_states_quasistatic(links_states)
    bagfile_name = pathlib.Path(tempfile.NamedTemporaryFile().name)
    return save_test_scene_given_name(joint_state, links_states, bagfile_name, force=True)


def make_links_states_quasistatic(links_states: LinkStates):
    for twist in links_states.twist:
        twist.linear.x = 0
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = 0
