cmake_minimum_required(VERSION 3.0.2)
project(merp)

set(Torch_DIR /home/peter/catkin_ws/src/venv/lib/python3.8/site-packages/torch/share/cmake/Torch)
set(CMAKE_CXX_STANDARD 20)

find_package(catkin REQUIRED COMPONENTS rospy)
find_package(Torch REQUIRED)

catkin_python_setup()

catkin_package()

add_executable(torch_test src/torch_test.cpp)
target_link_libraries(torch_test ${TORCH_LIBRARIES})