#pragma once

#include <thread>

#include <ros/callback_queue.h>
#include <tf/transform_listener.h>
#include <ros/ros.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

#include <gazebo/common/Events.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>

namespace gazebo
{
class GazeboRosTfPlugin : public WorldPlugin
{
 public:
  ~GazeboRosTfPlugin() override;

  void Load(physics::WorldPtr world, sdf::ElementPtr sdf) override;

 private:
  std::unique_ptr<ros::NodeHandle> ph_;
  ros::CallbackQueue queue_;
  std::thread callback_queue_thread_;
  std::thread periodic_event_thread_;

  tf2_ros::TransformBroadcaster tb_;
  tf2_ros::StaticTransformBroadcaster stb_;
  tf::TransformListener tf_listener_;

  event::ConnectionPtr update_connection_;
  physics::WorldPtr world_;
  std::vector<std::string> model_names_;
  std::string frame_id_ = "world";
  std::string robot_root_frame_id_ = "base_link";
  ros::Time last_tf_update_;

  void PrivateQueueThread();

  void PeriodicUpdate();

  std::string get_frame_id(const std::string &model_name, const std::string &link_name) const;
};
}  // namespace gazebo