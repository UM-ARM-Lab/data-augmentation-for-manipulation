cmake_minimum_required(VERSION 3.0.0)
project(merrrt_visualization)

find_package(Qt5 COMPONENTS Core Widgets REQUIRED)
find_package(catkin REQUIRED COMPONENTS roscpp peter_msgs arc_utilities)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(CMAKE_CXX_STANDARD 17)

catkin_python_setup()

catkin_package()

catkin_package(
   INCLUDE_DIRS include
   CATKIN_DEPENDS roscpp peter_msgs arc_utilities
   )

set(rviz_animation_controller_src
        src/merrrt_visualization/rviz_animation_controller.cpp
        include/merrrt_visualization/rviz_animation_controller.h)
add_library(rviz_animation_controller ${rviz_animation_controller_src})
target_include_directories(rviz_animation_controller PUBLIC include ${catkin_INCLUDE_DIRS})
target_link_libraries(rviz_animation_controller PUBLIC Qt5::Core Qt5::Widgets ${catkin_LIBRARIES} uuid)
add_dependencies(rviz_animation_controller ${catkin_EXPORTED_TARGETS})

add_library(merrrt_widget src/merrrt_visualization/merrrt_widget.cpp include/merrrt_visualization/merrrt_widget.h)
target_include_directories(merrrt_widget PUBLIC include ${catkin_INCLUDE_DIRS})
target_link_libraries(merrrt_widget PUBLIC Qt5::Core Qt5::Widgets ${catkin_LIBRARIES})
add_dependencies(merrrt_widget ${catkin_EXPORTED_TARGETS})

add_library(color_filter_widget src/merrrt_visualization/color_filter_widget.cpp include/merrrt_visualization/color_filter_widget.h)
target_include_directories(color_filter_widget PUBLIC include ${catkin_INCLUDE_DIRS})
target_link_libraries(color_filter_widget PUBLIC Qt5::Core Qt5::Widgets ${catkin_LIBRARIES})
add_dependencies(color_filter_widget ${catkin_EXPORTED_TARGETS})

add_library(manual_path_widget src/merrrt_visualization/manual_path_widget.cpp include/merrrt_visualization/manual_path_widget.h)
target_include_directories(manual_path_widget PUBLIC include ${catkin_INCLUDE_DIRS})
target_link_libraries(manual_path_widget PUBLIC Qt5::Core Qt5::Widgets ${catkin_LIBRARIES})
add_dependencies(manual_path_widget ${catkin_EXPORTED_TARGETS})
