#!/usr/bin/env python
import rospy
from link_bot_pycommon.dual_arm_sim_rope_scenario import SimDualArmRopeScenario


def main():
    rospy.init_node("val_grasp_rope")
    s = SimDualArmRopeScenario('hdt_michigan')
    s.on_before_get_state_or_execute_action()

    s.robot.plan_to_joint_config('both_arms', 'home')
    s.open_grippers_if_not_grasping()
    s.grasp_rope_endpoints()
    s.robot.open_left_gripper()
    s.robot.open_right_gripper()


if __name__ == "__main__":
    main()
