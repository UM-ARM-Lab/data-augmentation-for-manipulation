#pragma once

#include <memory>

#include <geometry_msgs/Pose.h>
#include <peter_msgs/GetPosition3D.h>
#include <peter_msgs/Position3DAction.h>
#include <peter_msgs/Position3DEnable.h>
#include <peter_msgs/RegisterPosition3DController.h>
#include <peter_msgs/UnregisterPosition3DController.h>
#include <peter_msgs/Position3DStop.h>
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <std_msgs/String.h>

#include <functional>
#include <gazebo/common/Events.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/common/Time.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/transport/TransportTypes.hh>

#include <link_bot_gazebo/base_link_position_controller.h>

namespace gazebo
{
class Position3dPlugin : public WorldPlugin
{
 public:
  ~Position3dPlugin() override;

  void Load(physics::WorldPtr world, sdf::ElementPtr sdf) override;

  bool OnStop(peter_msgs::Position3DStopRequest &req, peter_msgs::Position3DStopResponse &res);

  bool OnEnable(peter_msgs::Position3DEnableRequest &req, peter_msgs::Position3DEnableResponse &res);

  bool OnSet(peter_msgs::Position3DActionRequest &req, peter_msgs::Position3DActionResponse &res);

  bool OnRegister(peter_msgs::RegisterPosition3DControllerRequest &req,
                  peter_msgs::RegisterPosition3DControllerResponse &res);

  bool OnUnregister(peter_msgs::UnregisterPosition3DControllerRequest &req,
                    peter_msgs::UnregisterPosition3DControllerResponse &res);

  bool GetPos(peter_msgs::GetPosition3DRequest &req, peter_msgs::GetPosition3DResponse &res);

 private:
  void QueueThread();

  void PrivateQueueThread();

  void OnUpdate();

  // unique_ptr because base class is abstract
  std::unordered_map<std::string, std::unique_ptr<BaseLinkPositionController>> controllers_map_;


  physics::WorldPtr world_;
  event::ConnectionPtr update_connection_;
  std::unique_ptr<ros::NodeHandle> private_ros_node_;
  ros::NodeHandle ros_node_;
  ros::CallbackQueue queue_;
  ros::CallbackQueue private_queue_;
  std::thread ros_queue_thread_;
  std::thread private_ros_queue_thread_;
  ros::ServiceServer register_service_;
  ros::ServiceServer unregister_service_;
  ros::ServiceServer enable_service_;
  ros::ServiceServer set_service_;
  ros::ServiceServer stop_service_;
  ros::ServiceServer get_position_service_;

  void CreateServices();
};

}  // namespace gazebo
