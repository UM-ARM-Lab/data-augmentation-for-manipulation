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

add_executable(interactive_icp src/interactive_icp.cpp)
target_link_libraries(interactive_icp ${catkin_LIBRARIES})
target_include_directories(interactive_icp PUBLIC ${catkin_INCLUDE_DIRS})

add_executable(merge_pointclouds src/merge_pointclouds.cpp)
target_link_libraries(merge_pointclouds ${catkin_LIBRARIES})
target_include_directories(merge_pointclouds PUBLIC ${catkin_INCLUDE_DIRS})
