cmake_minimum_required(VERSION 3.0.2)
project(arm_moveit)

set(CMAKE_CXX_STANDARD 17)
set(CATKIN_DEPENDS
  moveit_commander
  moveit_core
  moveit_visual_tools
  pybind11_catkin
  roscpp
)

find_package(catkin REQUIRED COMPONENTS ${CATKIN_DEPENDS})

catkin_python_setup()

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES arm_moveit robot_points_generator
  CATKIN_DEPENDS ${CATKIN_DEPENDS}
)

set(FLAGS -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror=return-type -Werror=missing-field-initializers)

add_library(robot_points_generator src/robot_points_generator.cpp)
target_include_directories(robot_points_generator PUBLIC include)
target_include_directories(robot_points_generator SYSTEM PUBLIC ${catkin_INCLUDE_DIRS})
target_link_libraries(robot_points_generator PUBLIC ${catkin_LIBRARIES})
target_compile_options(robot_points_generator PUBLIC ${FLAGS})

add_executable(robot_points_generator_node src/robot_points_generator_node.cpp)
target_link_libraries(robot_points_generator_node PUBLIC ${catkin_LIBRARIES} robot_points_generator)

pybind_add_module(pyrobot_points_generator src/bindings.cpp)
target_link_libraries(pyrobot_points_generator PUBLIC robot_points_generator)
