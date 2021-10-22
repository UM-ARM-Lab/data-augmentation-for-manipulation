import argparse
import pathlib
from copy import deepcopy
from typing import Callable

from arc_utilities import ros_init
from link_bot_data.dataset_utils import replaced_true_with_predicted
from link_bot_planning.my_planner import LoggingTree
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from link_bot_pycommon.serialization import load_gzipped_pickle
from merrrt_visualization.rviz_animation_controller import RvizAnimationController


def walk(parent: LoggingTree):
    for child in parent.children:
        yield parent, child
        yield from walk(child)


@ros_init.with_ros("viz_graph")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('graph', type=pathlib.Path)
    parser.add_argument('viz_type', choices=['full_tree', 'plans_to_goal', 'actual_plans_to_goal'])

    args = parser.parse_args()

    scenario = get_scenario("dual_arm_rope_sim_val_with_robot_feasibility_checking")

    graph_data = load_gzipped_pickle(args.graph)
    g = graph_data['graph']
    goal = graph_data['goal']
    params = graph_data['params']
    goal_threshold = 0.045

    def actual_reached_goal(node):
        d_to_goal = scenario.distance_to_goal(node, goal).numpy()
        if d_to_goal < goal_threshold:
            return True

    def predicted_reached_goal(node):
        predicted = replaced_true_with_predicted(node.state)
        d_to_goal = scenario.distance_to_goal(predicted, goal).numpy()
        if d_to_goal < goal_threshold:
            return True

    if args.viz_type == 'full_tree':
        anim_full_tree(g, scenario)
    elif args.viz_type == "plans_to_goal":
        p = extract_path(g, predicted_reached_goal)
        viz_path(scenario, p, goal, goal_threshold)
    elif args.viz_type == "actual_plans_to_goal":
        p = extract_path(g, actual_reached_goal)
        viz_path(scenario, p, goal, goal_threshold)
    else:
        raise NotImplementedError(args.viz_type)


def viz_path(scenario, path, goal, goal_threshold):
    scenario.plot_goal_rviz(goal, goal_threshold)
    anim = RvizAnimationController(n_time_steps=len(path))
    while not anim.done:
        t = anim.t()
        node = path[t]
        s_t = node.state

        scenario.plot_state_rviz(s_t)

        anim.step()


def extract_path(g: LoggingTree, cond: Callable):
    def _exatract_path(parent: LoggingTree, path=[]):
        path.append(parent)
        if cond(parent):
            return path

        for child in parent.children:
            final_path = _exatract_path(child, path)
            if final_path is not None:
                return final_path

        path.pop()

    return _exatract_path(g)


def anim_full_tree(g: LoggingTree, scenario: ScenarioWithVisualization):
    tree_flat = list(walk(g))
    anim = RvizAnimationController(n_time_steps=len(tree_flat))
    while not anim.done:
        i = anim.t()
        parent, child = tree_flat[i]
        scenario.plot_state_rviz(parent.state, label='before', idx=2 * i)
        scenario.plot_state_rviz(child.state, label='after', idx=2 * i + 1)
        actual = child.state
        predicted = replaced_true_with_predicted(child.state)
        label = scenario.compute_label(actual, predicted, labeling_params={'threshold': 0.05})
        error = scenario.classifier_distance(actual, predicted)
        scenario.plot_error_rviz(error)
        scenario.plot_is_close(label)
        if parent.action is not None:
            state_for_action = deepcopy(parent.state)
            state_for_action.pop("left_gripper")
            state_for_action.pop("right_gripper")
            state_for_action.pop("rope")
            scenario.plot_action_rviz(state_for_action, child.action, label='graph', idx=2 * i)
        anim.step()


if __name__ == '__main__':
    main()
