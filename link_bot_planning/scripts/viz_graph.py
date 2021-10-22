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


class StateActionTree:
    def __init__(self, state=None, action=None):
        self.state = state
        self.action = action
        self.children = []


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
        d_to_goal = scenario.distance_to_goal(node.state, goal).numpy()
        return d_to_goal < goal_threshold

    def predicted_reached_goal(node):
        predicted = replaced_true_with_predicted(node.state)
        d_to_goal = scenario.distance_to_goal(predicted, goal).numpy()
        return d_to_goal < goal_threshold

    if args.viz_type == 'full_tree':
        anim_full_tree(g, scenario)
    elif args.viz_type == "plans_to_goal":
        paths = extract_paths(g, predicted_reached_goal)
        viz_paths(scenario, paths, goal, goal_threshold)
    elif args.viz_type == "actual_plans_to_goal":
        paths = extract_paths(g, actual_reached_goal)
        viz_paths(scenario, paths, goal, goal_threshold)
    else:
        raise NotImplementedError(args.viz_type)


def viz_path(scenario, path, goal, goal_threshold):
    scenario.plot_goal_rviz(goal, goal_threshold)
    anim = RvizAnimationController(n_time_steps=len(path))
    while not anim.done:
        t = anim.t()
        node = path[t]

        s_t = node['state']
        scenario.plot_state_rviz(s_t)
        if t < anim.max_t:
            next_node = path[t + 1]
            a_t = next_node['action']
            if a_t is not None:
                state_for_action = deepcopy(s_t)
                state_for_action.pop("left_gripper")
                state_for_action.pop("right_gripper")
                state_for_action.pop("rope")
                scenario.plot_action_rviz(state_for_action, a_t)

        anim.step()


def viz_paths(scenario, paths, goal, goal_threshold):
    if len(paths) == 0:
        return

    anim = RvizAnimationController(n_time_steps=len(paths), ns='trajs')
    while not anim.done:
        t = anim.t()
        path = paths[t]
        viz_path(scenario, path, goal, goal_threshold)
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


def tree_to_paths(parent: StateActionTree, path=[]):
    path.append({'state': parent.state, 'action': parent.action})

    if len(parent.children) == 0:
        yield path

    for child in parent.children:
        yield from tree_to_paths(child, path)

    path.pop()


def trim_tree(parent: LoggingTree, other_parent: StateActionTree, cond):
    new = StateActionTree(parent.state, parent.action)
    other_parent.children.append(new)

    if cond(parent):
        return

    for child in parent.children:
        trim_tree(child, new, cond)

    # if not a goal and no children left, remove
    if not cond(parent) and len(new.children) == 0:
        other_parent.children.pop()


def extract_paths(g: LoggingTree, cond: Callable):
    trimmed_tree = StateActionTree()
    trim_tree(g, trimmed_tree, cond)
    if len(trimmed_tree.children) == 0:
        print("No paths!")
        return []
    return list(tree_to_paths(trimmed_tree.children[0]))


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
