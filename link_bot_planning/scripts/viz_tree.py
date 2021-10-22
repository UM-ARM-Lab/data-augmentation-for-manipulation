import argparse
import pathlib

from arc_utilities import ros_init
from link_bot_planning.tree_utils import viz_paths, extract_paths, anim_full_tree, make_actual_reached_goal, \
    make_predicted_reached_goal
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.serialization import load_gzipped_pickle


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

    if args.viz_type == 'full_tree':
        anim_full_tree(g, scenario)
    elif args.viz_type == "plans_to_goal":
        paths = extract_paths(g, make_predicted_reached_goal(scenario, goal, goal_threshold))
        viz_paths(scenario, paths, goal, goal_threshold)
    elif args.viz_type == "actual_plans_to_goal":
        paths = extract_paths(g, make_actual_reached_goal(scenario, goal, goal_threshold))
        viz_paths(scenario, paths, goal, goal_threshold)
    else:
        raise NotImplementedError(args.viz_type)


if __name__ == '__main__':
    main()
