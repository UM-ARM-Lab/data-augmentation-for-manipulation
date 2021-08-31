cmake_minimum_required(VERSION 3.10)
project(link_bot_perception)

find_package(catkin REQUIRED rospy link_bot_data cdcpd)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES link_bot_perception
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

catkin_python_setup()
