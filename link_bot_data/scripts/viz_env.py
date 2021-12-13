from link_bot_pycommon.get_scenario import get_scenario
from arc_utilities import ros_init


def main():
    ros_init.rospy_and_cpp_init("viz_env")
    s = get_scenario("real_val_with_robot_feasibility_checking")
    params = {
        'extent': [-0.5, 0.5, 0, 1, -0.4, 0.6],
        'res':    0.02,
    }
    s.on_before_get_state_or_execute_action()
    e = s.get_environment(params)
    for i in range(3):
        s.plot_environment_rviz(e)
    print()


if __name__ == '__main__':
    main()
