#!/usr/bin/env python3

from link_bot_pycommon.dual_floating_gripper_scenario import DualFloatingGripperRopeScenario
from time import sleep
import numpy as np
import rospy
from link_bot_pycommon.dual_floating_gripper_scenario import sample_rope, sample_rope_grippers, sample_rope_and_grippers

rospy.init_node("test_random_rope_config")
sc = DualFloatingGripperRopeScenario()

p = [0, 0, 0]
g1 = [-0.1, 0, 0.4]
g2 = [0.1, 0, 0.4]

rng = np.random.RandomState(0)
for i in range(100):
    rope = sample_rope_and_grippers(rng, g1, g2, p, DualFloatingGripperRopeScenario.n_links, kd=0.05)
    gripper1 = g1
    gripper2 = g2

    # rope = sample_rope_grippers(rng, g1, g2, DualFloatingGripperRopeScenario.n_links)
    # gripper1= g1
    # gripper2= g2

    # rope = sample_rope(rng, p, DualFloatingGripperRopeScenario.n_links, kd=0.05)
    # gripper1= rope[-1]
    # gripper2= rope[0]

    state = {
        'gripper1': gripper1,
        'gripper2': gripper2,
        'link_bot': rope.flatten(),
    }
    sc.plot_state_rviz(state, label='test')
    sleep(0.5)
