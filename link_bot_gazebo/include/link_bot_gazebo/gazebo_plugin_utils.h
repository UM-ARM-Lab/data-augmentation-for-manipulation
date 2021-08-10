#pragma once

#include <geometry_msgs/Point.h>

#include <gazebo/physics/Joint.hh>
#include <gazebo/physics/Link.hh>
#include <gazebo/physics/Model.hh>
#include <gazebo/physics/World.hh>
#include <string>

gazebo::physics::LinkPtr GetLink(char const *plugin_name, gazebo::physics::ModelPtr model, std::string const link_name);

gazebo::physics::JointPtr GetJoint(char const *plugin_name, gazebo::physics::ModelPtr model,
                                   std::string const link_name);

gazebo::physics::LinkPtr GetLink(char const *plugin_name, gazebo::physics::WorldPtr world,
                                 std::string const scoped_link_name);

geometry_msgs::Point ign_vector_3d_to_point(ignition::math::Vector3d const &pos);

ignition::math::Vector3d point_to_ign_vector_3d(geometry_msgs::Point const &pos);

ignition::math::Pose3d pose_to_ign_pose(geometry_msgs::Pose const &pose);