#!/usr/bin/env python
from arc_utilities import ros_init
from link_bot_pycommon.dual_arm_sim_rope_scenario import SimDualArmRopeScenario


@ros_init.with_ros("val_grasp_rope")
def main():
    s = SimDualArmRopeScenario('hdt_michigan', params={'rope_name': 'rope_3d_alt'})
    s.on_before_get_state_or_execute_action()

    s.robot.plan_to_joint_config('both_arms', 'home')
    s.open_grippers_if_not_grasping()
    s.grasp_rope_endpoints()


if __name__ == "__main__":
    main()
