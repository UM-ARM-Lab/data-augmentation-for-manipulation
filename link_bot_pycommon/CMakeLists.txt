cmake_minimum_required(VERSION 2.8.3)
project(link_bot_pycommon)

find_package(catkin REQUIRED COMPONENTS
        rospy
        std_msgs
        arm_gazebo_msgs)

catkin_package(
        CATKIN_DEPENDS std_msgs rospy arm_gazebo_msgs
)

catkin_python_setup()
