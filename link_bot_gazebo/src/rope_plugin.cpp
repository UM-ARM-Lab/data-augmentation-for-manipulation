#include "rope_plugin.h"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <sstream>

#include <boost/regex.hpp>
#include <gazebo/common/Time.hh>
#include <ros/console.h>
#include <std_msgs/Float64.h>
#include <visualization_msgs/MarkerArray.h>

#include <arc_utilities/enumerate.h>
#include <link_bot_gazebo/gazebo_plugin_utils.h>

/**
 * This plugin offers the following ROS API
 * Services:
 * - set_rope_state
 *   - type: peter_msgs::SetRopeState
 * - get_rope_state
 *   - type: peter_msgs::GetRopeState
 * - rope_overstretched
 *   - type: peter_msgs::GetOverstretching
 */

auto is_ptr_valid = [](auto p) { return p; };

namespace gazebo
{
auto constexpr PLUGIN_NAME{"RopePlugin"};

void RopePlugin::Load(physics::ModelPtr const parent, sdf::ElementPtr const sdf)
{
  model_ = parent;

  // Make sure the ROS node for Gazebo has already been initialized
  if (!ros::isInitialized())
  {
    auto argc = 0;
    char **argv = nullptr;
    ros::init(argc, argv, "rope_plugin", ros::init_options::NoSigintHandler);
  }

  n_links_ = model_->GetLinks().size() - 2;
  for (auto link_idx = 1u; link_idx <= n_links_; ++link_idx)
  {
    auto const link_name = "rope_link_" + std::to_string(link_idx);
    auto const rope_link = GetLink(PLUGIN_NAME, model_, link_name);
    if (rope_link)
    {
      rope_links_.push_back(rope_link);
    }
  }
  left_gripper_ = GetLink(PLUGIN_NAME, model_, "left_gripper");
  right_gripper_ = GetLink(PLUGIN_NAME, model_, "right_gripper");

  if (std::all_of(rope_links_.cbegin(), rope_links_.cend(), is_ptr_valid))
  {
    rest_distance_ = 0.0;
    for (auto link_idx1 = 0u; link_idx1 < n_links_ - 1; ++link_idx1)
    {
      auto const link_idx2 = link_idx1 + 1;
      auto const rope_link1_ = rope_links_[link_idx1];
      auto const rope_link2_ = rope_links_[link_idx2];
      auto const d = (rope_link1_->WorldPose().Pos() - rope_link2_->WorldPose().Pos()).Length();
      rest_distance_ += d;
      ROS_DEBUG_STREAM_NAMED(PLUGIN_NAME, "distance between link " << link_idx1 << " and " << link_idx2 << " is " << d);
    }
    rest_distance_ = rest_distance_ / static_cast<double>(n_links_ - 1);
  }
  ROS_DEBUG_STREAM_NAMED(PLUGIN_NAME, "rest distance " << rest_distance_);

  auto set_state_bind = [this](auto &&req, auto &&res) { return SetRopeState(req, res); };
  auto set_state_so = ros::AdvertiseServiceOptions::create<peter_msgs::SetRopeState>("set_rope_state", set_state_bind,
                                                                                     ros::VoidPtr(), &queue_);

  auto get_state_bind = [this](auto &&req, auto &&res) { return GetRopeState(req, res); };
  auto get_state_so = ros::AdvertiseServiceOptions::create<peter_msgs::GetRopeState>("get_rope_state", get_state_bind,
                                                                                     ros::VoidPtr(), &queue_);

  auto overstretched_bind = [this](auto &&req, auto &&res) { return GetOverstretched(req, res); };
  auto overstretched_so = ros::AdvertiseServiceOptions::create<peter_msgs::GetOverstretching>(
      "rope_overstretched", overstretched_bind, ros::VoidPtr(), &queue_);

  ros_node_ = std::make_unique<ros::NodeHandle>(model_->GetScopedName());
  set_state_service_ = ros_node_->advertiseService(set_state_so);
  rope_overstretched_service_ = ros_node_->advertiseService(overstretched_so);
  get_state_service_ = ros_node_->advertiseService(get_state_so);
  overstretching_pub_ = ros_node_->advertise<std_msgs::Float64>("overstretching", 10);
  viz_pub_ = ros_node_->advertise<visualization_msgs::MarkerArray>("rope_viz", 10);

  ros_queue_thread_ = std::thread([this] { QueueThread(); });

  {
    if (sdf->HasElement("overstretching_factor"))
    {
      overstretching_factor_ = sdf->GetElement("overstretching_factor")->Get<double>();
    }

    if (!sdf->HasElement("num_links"))
    {
      printf("using default num_links=%u\n", num_links_);
    } else
    {
      num_links_ = sdf->GetElement("num_links")->Get<unsigned int>();
    }
  }
  ROS_INFO("Rope Plugin finished initializing!");

  update_conn_ = event::Events::ConnectWorldUpdateBegin(std::bind(&RopePlugin::OnUpdate, this));

  auto periodic_update_func = [this]
  {
    while (true)
    {
      PeriodicUpdate();
      ros::Duration(0.02).sleep();
    }
  };
  periodic_event_thread_ = std::thread(periodic_update_func);
}

void RopePlugin::OnUpdate()
{
  UpdateOverstretching();
}
void RopePlugin::PeriodicUpdate()
{
  visualization_msgs::MarkerArray rope_markers;
  visualization_msgs::Marker points_marker;
  points_marker.header.stamp = ros::Time::now();
  points_marker.header.frame_id = "world";
  points_marker.id = 0;
  points_marker.action = visualization_msgs::Marker::ADD;
  points_marker.type = visualization_msgs::Marker::SPHERE_LIST;
  points_marker.pose.orientation.w = 1;
  points_marker.color.r = 0.0;
  points_marker.color.g = 1.0;
  points_marker.color.b = 1.0;
  points_marker.color.a = 1.0;
  auto const s = 0.01;
  points_marker.scale.x = s;
  points_marker.scale.y = s;
  points_marker.scale.z = s;

  visualization_msgs::Marker line_marker;
  line_marker.header.stamp = ros::Time::now();
  line_marker.header.frame_id = "world";
  line_marker.id = 1;
  line_marker.action = visualization_msgs::Marker::ADD;
  line_marker.type = visualization_msgs::Marker::LINE_STRIP;
  line_marker.pose.orientation.w = 1;
  line_marker.color.r = 0.2;
  line_marker.color.g = 0.8;
  line_marker.color.b = 0.8;
  line_marker.color.a = 1.0;
  line_marker.scale.x = s;

  for (auto const &pair : enumerate(model_->GetLinks()))
  {
    auto const &[i, link] = pair;
    auto const name = link->GetName();
    boost::regex e(".*rope_link_\\d+");
    if (boost::regex_match(name, e))
    {
      geometry_msgs::Point pt;
      pt.x = link->WorldPose().Pos().X();
      pt.y = link->WorldPose().Pos().Y();
      pt.z = link->WorldPose().Pos().Z();
      points_marker.points.push_back(pt);
      line_marker.points.push_back(pt);
    }
  }

  rope_markers.markers.push_back(points_marker);
  rope_markers.markers.push_back(line_marker);

  viz_pub_.publish(rope_markers);
}
bool RopePlugin::SetRopeState(peter_msgs::SetRopeStateRequest &req, peter_msgs::SetRopeStateResponse &)
{
  for (const auto& pair : enumerate(model_->GetJoints()))
  {
    auto const &[i, joint] = pair;
    if (i < req.joint_angles_axis1.size())
    {
      joint->SetPosition(0, req.joint_angles_axis1[i]);
      joint->SetPosition(1, req.joint_angles_axis2[i]);
    }
  }
  if (left_gripper_ and right_gripper_)
  {
    left_gripper_->SetWorldPose({req.left_gripper.x, req.left_gripper.y, req.left_gripper.z, 0, 0, 0});
    right_gripper_->SetWorldPose({req.right_gripper.x, req.right_gripper.y, req.right_gripper.z, 0, 0, 0});
  } else
  {
    ROS_ERROR_STREAM("Tried to set link to pose but couldn't find the gripper links");
    ROS_ERROR_STREAM("Available link names are");
    for (auto const &l : model_->GetLinks())
    {
      ROS_ERROR_STREAM(l->GetName());
    }
  }

  return true;
}

bool RopePlugin::GetRopeState(peter_msgs::GetRopeStateRequest &, peter_msgs::GetRopeStateResponse &res)
{
  static peter_msgs::GetRopeStateResponse previous_res;

  for (auto const &joint : model_->GetJoints())
  {
    res.joint_angles_axis1.push_back(joint->Position(0));
    res.joint_angles_axis2.push_back(joint->Position(1));
  }
  for (auto const &pair : enumerate(model_->GetLinks()))
  {
    auto const &[i, link] = pair;
    auto const name = link->GetName();
    boost::regex e(".*rope_link_\\d+");
    if (boost::regex_match(name, e))
    {
      geometry_msgs::Point pt;
      pt.x = link->WorldPose().Pos().X();
      pt.y = link->WorldPose().Pos().Y();
      pt.z = link->WorldPose().Pos().Z();
      res.positions.emplace_back(pt);

      geometry_msgs::Point velocity;
      if (velocity_initialized_)
      {
        velocity.x = pt.x - previous_res.positions[i].x;
        velocity.y = pt.y - previous_res.positions[i].y;
        velocity.z = pt.z - previous_res.positions[i].z;
      } else
      {
        velocity.x = 0;
        velocity.y = 0;
        velocity.z = 0;
      }
      res.velocities.emplace_back(velocity);
    } else
    {
      // ROS_INFO_STREAM("skipping link with name " << name);
    }
  }
  res.model_pose.position.x = model_->WorldPose().Pos().X();
  res.model_pose.position.y = model_->WorldPose().Pos().Y();
  res.model_pose.position.z = model_->WorldPose().Pos().Z();
  res.model_pose.orientation.x = model_->WorldPose().Rot().X();
  res.model_pose.orientation.y = model_->WorldPose().Rot().Y();
  res.model_pose.orientation.z = model_->WorldPose().Rot().Z();
  res.model_pose.orientation.w = model_->WorldPose().Rot().W();

  previous_res = res;
  velocity_initialized_ = true;

  return true;
}


void RopePlugin::UpdateOverstretching()
{
  // check the distance between the position of rope_link_1 and gripper_1
  if (not std::all_of(rope_links_.cbegin(), rope_links_.cend(), is_ptr_valid))
  {
    return;
  }

  auto max_distance = 0.0;
  for (auto link_idx1 = 0u; link_idx1 < n_links_ - 1; ++link_idx1)
  {
    auto const link_idx2 = link_idx1 + 1;
    auto const rope_link1_ = rope_links_[link_idx1];
    auto const rope_link2_ = rope_links_[link_idx2];
    auto const d = (rope_link1_->WorldPose().Pos() - rope_link2_->WorldPose().Pos()).Length();
    max_distance = std::max(max_distance, d);
  }

  rope_overstretching_filter_.addSample(max_distance);
  auto const filtered_max_distance = rope_overstretching_filter_.getMedian();
  ROS_DEBUG_STREAM_NAMED(PLUGIN_NAME,
                         "max distance " << filtered_max_distance << " vs rest distance " << rest_distance_);

  auto const overstretched = filtered_max_distance > (rest_distance_ * overstretching_factor_);
  if (overstretched)
  {
    ROS_DEBUG_STREAM_THROTTLE_NAMED(1, PLUGIN_NAME, "overstretching detected!");
  }

  auto const magnitude = filtered_max_distance / rest_distance_;
  {
    std::lock_guard g(mutex_);
    overstretching_response_.overstretched = overstretched;
    overstretching_response_.magnitude = magnitude;
  }

  std_msgs::Float64 overstretching;
  overstretching.data = magnitude;
  overstretching_pub_.publish(overstretching);
}

bool RopePlugin::GetOverstretched(peter_msgs::GetOverstretchingRequest &req, peter_msgs::GetOverstretchingResponse &res)
{
  (void) req;  // unused
  {
    std::lock_guard g(mutex_);
    res.magnitude = overstretching_response_.magnitude;
    res.overstretched = overstretching_response_.overstretched;
  }
  return true;
}

void RopePlugin::QueueThread()
{
  double constexpr timeout = 0.01;
  while (ros_node_->ok())
  {
    queue_.callAvailable(ros::WallDuration(timeout));
  }
}

RopePlugin::~RopePlugin()
{
  queue_.clear();
  queue_.disable();
  ros_node_->shutdown();
  ros_queue_thread_.join();
}

// Register this plugin with the simulator
GZ_REGISTER_MODEL_PLUGIN(RopePlugin)
}  // namespace gazebo
