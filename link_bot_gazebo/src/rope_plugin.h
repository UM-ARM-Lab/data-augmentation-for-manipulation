#pragma once

#include <peter_msgs/GetOverstretching.h>
#include <peter_msgs/GetRopeState.h>
#include <peter_msgs/SetRopeState.h>
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <ros/subscribe_options.h>
#include <sensor_msgs/Joy.h>
#include <std_msgs/String.h>
#include <std_srvs/Empty.h>

#include <Eigen/Eigen>
#include <gazebo/common/common.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/sensors/sensors.hh>
#include <ignition/math.hh>
#include <mutex>
#include <sdf/sdf.hh>
#include <string>
#include <thread>

#include <link_bot_gazebo/median_filter.h>

namespace gazebo
{
class RopePlugin : public ModelPlugin
{
public:
  void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/) override;

  ~RopePlugin() override;

  bool SetRopeState(peter_msgs::SetRopeStateRequest &req, peter_msgs::SetRopeStateResponse &res);

  bool GetRopeState(peter_msgs::GetRopeStateRequest &req, peter_msgs::GetRopeStateResponse &res);

  bool GetOverstretched(peter_msgs::GetOverstretchingRequest &req, peter_msgs::GetOverstretchingResponse &res);

  void OnUpdate();

  void PeriodicUpdate();

  void UpdateOverstretching();

private:
  void QueueThread();

  event::ConnectionPtr update_conn_;
  physics::ModelPtr model_;
  physics::Link_V rope_links_;
  physics::LinkPtr left_gripper_;
  physics::LinkPtr right_gripper_;
  event::ConnectionPtr updateConnection_;
  double length_{ 0.0 };
  bool velocity_initialized_{false };
  double rest_distance_{0.0 };
  unsigned int num_links_{ 0U };
  std::unique_ptr<ros::NodeHandle> ros_node_;
  ros::ServiceServer set_state_service_;
  ros::ServiceServer rope_overstretched_service_;
  ros::ServiceServer get_state_service_;
  ros::Publisher overstretching_pub_;
  ros::Publisher viz_pub_;
  ros::CallbackQueue queue_;
  std::thread ros_queue_thread_;
  double overstretching_factor_{ 1.0 };
  unsigned long n_links_{1ul};
  MedianFilter<double, 100> rope_overstretching_filter_;
  std::mutex mutex_;
  peter_msgs::GetOverstretchingResponse overstretching_response_;
  std::thread periodic_event_thread_;
};
}  // namespace gazebo
