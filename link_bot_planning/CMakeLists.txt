cmake_minimum_required(VERSION 2.8.3)
cmake_policy(SET CMP0048 NEW)
project(link_bot_planning)

find_package(catkin REQUIRED COMPONENTS rospy ompl pybind11_catkin)

catkin_python_setup()

catkin_package(
        INCLUDE_DIRS include
        LIBRARIES link_bot_planning
        CATKIN_DEPENDS rospy std_msgs sensor_msgs geometry_msgs
        DEPENDS OMPL
)

# add_library(link_bot_planning src/floating_rope_ompl.cpp)
# target_include_directories(link_bot_planning SYSTEM PUBLIC include ${catkin_INCLUDE_DIRS} ${OMPL_INCLUDE_DIRS})
# target_link_libraries(link_bot_planning PUBLIC ${catkin_LIBRARIES} ${OMPL_LIBRARIES})