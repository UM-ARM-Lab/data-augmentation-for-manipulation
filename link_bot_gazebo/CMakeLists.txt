cmake_minimum_required(VERSION 3.5)
project(link_bot_gazebo)

find_package(
        catkin
        REQUIRED COMPONENTS
        rospy
        roscpp
        sdf_tools
        arm_video_recorder
        arc_utilities
        peter_msgs
        control_msgs
        moveit_msgs
        sensor_msgs
        trajectory_msgs
        tf2_ros
        victor_hardware_interface
        visualization_msgs
)

find_package(cnpy REQUIRED)
find_package(gazebo REQUIRED)


set(CMAKE_CXX_STANDARD 17)

catkin_python_setup()

catkin_package(
        LIBRARIES
        CATKIN_DEPENDS
        gazebo_msgs
        geometry_msgs
        message_runtime
        moveit_msgs
        arc_utilities
        arm_video_recorder
        roscpp
        sensor_msgs
        std_msgs
        std_srvs
        trajectory_msgs
        peter_msgs
        tf2_ros
        victor_hardware_interface
        visualization_msgs
)

link_directories(${GAZEBO_LIBRARY_DIRS})
set(FLAGS ${GAZEBO_CXX_FLAGS} -O3 -g -Wall -Wextra -Wshadow -Werror=return-type -Werror=missing-field-initializers)

function(gazebo_plugin target_name)
    set(sources ${ARGN})
    add_library(${target_name} SHARED ${sources})
    target_compile_options(${target_name} PRIVATE ${FLAGS})
    target_include_directories(${target_name} PUBLIC include)
    target_include_directories(${target_name} SYSTEM PUBLIC ${catkin_INCLUDE_DIRS} ${GAZEBO_INCLUDE_DIRS})
    target_link_libraries(${target_name} ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})
    add_dependencies(${target_name} ${catkin_EXPORTED_TARGETS})
endfunction()

gazebo_plugin(rope_plugin src/rope_plugin.cpp src/gazebo_plugin_utils.cpp)

gazebo_plugin(stepping_plugin src/stepping_plugin.cpp)

gazebo_plugin(collision_map_plugin src/collision_map_plugin.cpp)
target_link_libraries(collision_map_plugin cnpy::cnpy stdc++fs)

gazebo_plugin(position_3d_plugin
        src/position_3d_plugin.cpp
        src/base_link_position_controller.cpp
        src/link_position_3d_kinematic_controller.cpp
        src/link_position_3d_pid_controller.cpp
        src/gazebo_plugin_utils.cpp)

gazebo_plugin(gazebo_ros_tf_plugin src/gazebo_ros_tf_plugin.cpp)

gazebo_plugin(kinematic_robotiq_3f_gripper_plugin
        src/gazebo_plugin_utils.cpp
        src/kinematic_robotiq_3f_gripper_plugin.cpp)

#gazebo_plugin(rosparam_plugin src/rosparam_plugin.cpp)

#gazebo_plugin(arm_video_recorder_plugin src/arm_video_recorder_plugin.cpp)
#set(LIBAV avformat avcodec swresample swscale avutil)
#target_link_libraries(arm_video_recorder_plugin CameraPlugin ${LIBAV})
