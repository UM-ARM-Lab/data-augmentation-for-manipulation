#!/usr/bin/env python3
import argparse
import pathlib
import pickle

import rospy
from arc_utilities import ros_init
from arm_robots.hdt_michigan import Val
from link_bot_gazebo.gazebo_services import GazeboServices
from link_bot_planning.planning_evaluation import load_planner_params
from link_bot_pycommon.get_scenario import get_scenario


@ros_init.with_ros("check_gripper_rotation")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('rope_name')

    args = parser.parse_args()

    s = get_scenario("dual_arm_rope_sim_val_with_robot_feasibility_checking", {'rope_name': args.rope_name})

    service_provider = GazeboServices()
    bagfile_name = pathlib.Path("test_scenes/car4_alt/scene_0000.bag")
    planner_params = load_planner_params(pathlib.Path("planner_configs/val_car/random_accept.hjson"))
    s.on_before_get_state_or_execute_action()
    val: Val = s.robot

    def _move():
        val.follow_jacobian_to_position('both_arms', ['left_tool', 'right_tool'], [
            [[0.95, 0.05, 1.0]],
            [[0.85, -0.05, 0.6]]
        ])
        rospy.sleep(2)

    def _hack():
        s.restore_from_bag(service_provider, planner_params, bagfile_name)
        val.store_current_tool_orientations(['left_tool', 'right_tool'])
        _move()

    _hack()

    s.restore_from_bag(service_provider, planner_params, bagfile_name)
    val.store_current_tool_orientations(['left_tool', 'right_tool'])
    rospy.sleep(2)

    before_state1 = s.get_state()
    s.plot_state_rviz(before_state1, label='before1', color='r')

    # move
    _move()

    after_state1 = s.get_state()
    s.plot_state_rviz(after_state1, label='after1', color='m')

    # now go back to the state state except the rope gripper will be rotated a little
    s.restore_from_bag(service_provider, planner_params, bagfile_name)
    val.store_current_tool_orientations(['left_tool', 'right_tool'])
    rospy.sleep(2)
    s.detach_rope_from_gripper('left_gripper')
    rospy.sleep(20)
    s.grasp_rope_endpoints()

    before_state2 = s.get_state()
    s.plot_state_rviz(before_state2, label='before2', color='c')

    # them make the same motion again
    _move()

    # get state
    after_state2 = s.get_state()
    s.plot_state_rviz(after_state2, label='after2', color='b')

    print(s.classifier_distance(after_state1, after_state2))

    print("done")


if __name__ == '__main__':
    main()
