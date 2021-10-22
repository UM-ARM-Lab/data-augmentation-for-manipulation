import argparse
import pathlib
from copy import deepcopy

from arc_utilities import ros_init
from link_bot_planning.my_planner import LoggingTree
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.serialization import load_gzipped_pickle, dump_gzipped_pickle
from merrrt_visualization.rviz_animation_controller import RvizSimpleStepper


@ros_init.with_ros("viz_graph")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('graph', type=pathlib.Path)

    args = parser.parse_args()

    scenario = get_scenario("dual_arm_rope_sim_val_with_robot_feasibility_checking")

    graph_data = load_gzipped_pickle(args.graph)
    g = graph_data['graph']

    stepper = RvizSimpleStepper()

    def walk(parent: LoggingTree):
        for child in parent.children:
            yield parent, child
            walk(child)

    for parent, child in walk(g):
        scenario.plot_state_rviz(parent.state, label='before')
        scenario.plot_state_rviz(child.state, label='after')
        if parent.action is not None:
            state_for_action = deepcopy(parent.state)
            state_for_action.pop("left_gripper")
            state_for_action.pop("right_gripper")
            state_for_action.pop("rope")
            scenario.plot_action_rviz(state_for_action, c.action, label='graph')
            stepper.step()

    dump_gzipped_pickle(graph_data, args.graph)


if __name__ == '__main__':
    main()
