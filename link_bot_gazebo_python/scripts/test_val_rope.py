#!/usr/bin/env python

import rospy
from arm_robots.hdt_michigan import Val
from link_bot_pycommon.dual_arm_sim_rope_scenario import SimValDualArmRopeScenario

rospy.init_node("test_val_rope")

val = Val()
s = SimValDualArmRopeScenario()
s.register_fake_grasping()
s.detach_rope_from_grippers()
s.make_rope_endpoints_follow_gripper()
