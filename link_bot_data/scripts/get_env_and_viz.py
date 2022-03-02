#!/usr/bin/env python
from link_bot_pycommon.get_scenario import get_scenario
import rospy
from arc_utilities import ros_init
from link_bot_pycommon.grid_utils_np import extent_res_to_origin_point


@ros_init.with_ros("get_env_and_viz")
def main():
    s = get_scenario("dual_arm_rope_sim_val_with_robot_feasibility_checking")
    s.on_before_get_state_or_execute_action()

    e = s.get_environment({'extent': [-0.3, 0.0, 0.3, 0.6, -0.1, 0.2], 'res': 0.02})

    s.plot_environment_rviz(e)
    rospy.sleep(1.0)


if __name__ == '__main__':
    main()
