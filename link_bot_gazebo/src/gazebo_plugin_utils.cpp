#include <geometry_msgs/Pose.h>
#include <link_bot_gazebo/gazebo_plugin_utils.h>
#include <ros/ros.h>

gazebo::physics::LinkPtr GetLink(char const *plugin_name, gazebo::physics::ModelPtr model,
                                 std::string const link_name) {
  auto const scoped_model_name = model->GetScopedName();
  auto const full_link_name = scoped_model_name + "::" + link_name;
  gazebo::physics::LinkPtr link = model->GetLink(full_link_name);
  if (!link) {
    ROS_ERROR_STREAM_NAMED(plugin_name, "No link " << full_link_name << " found");
    ROS_WARN_STREAM_NAMED(plugin_name, "Available links are:");
    for (const auto &l : model->GetLinks()) {
      ROS_WARN_STREAM_NAMED(plugin_name, l->GetName());
    }
  }
  return link;
}

gazebo::physics::JointPtr GetJoint(char const *plugin_name, gazebo::physics::ModelPtr model,
                                   std::string const joint_name) {
  auto const scoped_model_name = model->GetScopedName();
  auto const full_joint_name = scoped_model_name + "::" + joint_name;
  gazebo::physics::JointPtr joint = model->GetJoint(full_joint_name);
  if (!joint) {
    ROS_ERROR_STREAM_NAMED(plugin_name, "No joint " << full_joint_name << " found");
    ROS_WARN_STREAM_NAMED(plugin_name, "Available joints are:");
    for (const auto &j : model->GetJoints()) {
      ROS_WARN_STREAM_NAMED(plugin_name, j->GetName());
    }
  }
  return joint;
}

gazebo::physics::LinkPtr GetLink(char const *plugin_name, gazebo::physics::WorldPtr world,
                                 std::string const scoped_link_name) {
  std::vector<std::string> possible_names;
  for (auto const &model : world->Models()) {
    for (auto const &link : model->GetLinks()) {
      if (link->GetScopedName() == scoped_link_name) {
        return link;
      } else {
        possible_names.emplace_back(link->GetScopedName());
      }
    }
  }
  ROS_ERROR_STREAM_NAMED(plugin_name, "No link " << scoped_link_name << " found");
  ROS_WARN_STREAM_NAMED(plugin_name, "Available joints are:");
  for (auto const &n : possible_names) {
    ROS_WARN_STREAM_NAMED(plugin_name, n);
  }
  return nullptr;
}

geometry_msgs::Point ign_vector_3d_to_point(ignition::math::Vector3d const &pos) {
  geometry_msgs::Point point;
  point.x = pos.X();
  point.y = pos.Y();
  point.z = pos.Z();
  return point;
}

ignition::math::Vector3d point_to_ign_vector_3d(geometry_msgs::Point const &pos) { return {pos.x, pos.y, pos.z}; }

ignition::math::Pose3d pose_to_ign_pose(geometry_msgs::Pose const &pose) {
  return {pose.position.x,    pose.position.y,    pose.position.z,   pose.orientation.w,
          pose.orientation.x, pose.orientation.y, pose.orientation.z};
}
