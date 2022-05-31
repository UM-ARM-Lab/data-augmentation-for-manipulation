#include "position_3d_plugin.h"

#include <link_bot_gazebo/gazebo_plugin_utils.h>
#include <link_bot_gazebo/kinematic_robotiq_3f_gripper_plugin.h>
#include <link_bot_gazebo/link_position_3d_kinematic_controller.h>
#include <link_bot_gazebo/link_position_3d_pid_controller.h>
#include <ros/console.h>

#include <link_bot_gazebo/mymath.hpp>

#define create_service_options(type, name, bind) \
  ros::AdvertiseServiceOptions::create<type>(name, bind, ros::VoidPtr(), &queue_)

constexpr static auto const PLUGIN_NAME = "position_3d_plugin";

namespace gazebo {
GZ_REGISTER_WORLD_PLUGIN(Position3dPlugin)

Position3dPlugin::~Position3dPlugin() {
  queue_.clear();
  queue_.disable();
  ros_node_.shutdown();
  private_ros_node_->shutdown();
  ros_queue_thread_.join();
  private_ros_queue_thread_.join();
}

void Position3dPlugin::Load(physics::WorldPtr world, sdf::ElementPtr sdf) {
  world_ = world;

  if (!ros::isInitialized()) {
    ROS_FATAL_STREAM("A ROS node for Gazebo has not been initialized, unable to load plugin. ");
    return;
  }

  CreateServices();

  // FIXME: none of this code is thread-safe
}

void Position3dPlugin::CreateServices() {
  private_ros_node_ = std::make_unique<ros::NodeHandle>("position_3d_plugin");

  {
    auto stop_bind = [this](peter_msgs::Position3DStopRequest &req, peter_msgs::Position3DStopResponse &res) {
      return OnStop(req, res);
    };
    auto stop_so = create_service_options(peter_msgs::Position3DStop, "stop", stop_bind);
    stop_service_ = private_ros_node_->advertiseService(stop_so);
  }

  {
    auto register_bind = [this](peter_msgs::RegisterPosition3DControllerRequest &req,
                                peter_msgs::RegisterPosition3DControllerResponse &res) { return OnRegister(req, res); };
    auto register_so = create_service_options(peter_msgs::RegisterPosition3DController, "register", register_bind);
    register_service_ = private_ros_node_->advertiseService(register_so);
  }

  {
    auto unregister_bind = [this](peter_msgs::UnregisterPosition3DControllerRequest &req,
                                  peter_msgs::UnregisterPosition3DControllerResponse &res) {
      return OnUnregister(req, res);
    };
    auto unregister_so =
        create_service_options(peter_msgs::UnregisterPosition3DController, "unregister", unregister_bind);
    unregister_service_ = private_ros_node_->advertiseService(unregister_so);
  }

  {
    auto enable_bind = [this](peter_msgs::Position3DEnableRequest &req, peter_msgs::Position3DEnableResponse &res) {
      return OnEnable(req, res);
    };
    auto enable_so = create_service_options(peter_msgs::Position3DEnable, "enable", enable_bind);
    enable_service_ = private_ros_node_->advertiseService(enable_so);
  }

  {
    auto follow_bind = [this](peter_msgs::Position3DFollowRequest &req, peter_msgs::Position3DFollowResponse &res) {
      return OnFollow(req, res);
    };
    auto follow_so = create_service_options(peter_msgs::Position3DFollow, "follow", follow_bind);
    follow_service_ = private_ros_node_->advertiseService(follow_so);
  }

  {
    auto pos_set_bind = [this](peter_msgs::Position3DActionRequest &req, peter_msgs::Position3DActionResponse &res) {
      return OnSet(req, res);
    };
    auto pos_set_so = create_service_options(peter_msgs::Position3DAction, "position_set", pos_set_bind);
    position_set_service_ = private_ros_node_->advertiseService(pos_set_so);
  }

  {
    auto pose_set_bind = [this](peter_msgs::Pose3DActionRequest &req, peter_msgs::Pose3DActionResponse &res) {
      return OnSetPose(req, res);
    };
    auto pose_set_so = create_service_options(peter_msgs::Pose3DAction, "pose_set", pose_set_bind);
    pose_set_service_ = private_ros_node_->advertiseService(pose_set_so);
  }

  {
    auto pos_list_bind = [this](peter_msgs::Position3DListRequest &req, peter_msgs::Position3DListResponse &res) {
      return OnList(req, res);
    };
    auto pos_list_so = create_service_options(peter_msgs::Position3DList, "list", pos_list_bind);
    list_service_ = private_ros_node_->advertiseService(pos_list_so);
  }

  {
    auto pos_wait_bind = [this](peter_msgs::Position3DWaitRequest &req, peter_msgs::Position3DWaitResponse &res) {
      return OnWait(req, res);
    };
    auto pos_wait_so = create_service_options(peter_msgs::Position3DWait, "wait", pos_wait_bind);
    wait_service_ = private_ros_node_->advertiseService(pos_wait_so);
  }

  {
    auto pos_move_bind = [this](peter_msgs::Position3DActionRequest &req, peter_msgs::Position3DActionResponse &res) {
      return OnMove(req, res);
    };
    auto pos_move_so = create_service_options(peter_msgs::Position3DAction, "move", pos_move_bind);
    move_service_ = private_ros_node_->advertiseService(pos_move_so);
  }

  {
    auto get_pos_bind = [this](peter_msgs::GetPosition3DRequest &req, peter_msgs::GetPosition3DResponse &res) {
      return GetPos(req, res);
    };
    auto get_pos_so = create_service_options(peter_msgs::GetPosition3D, "get", get_pos_bind);
    get_position_service_ = private_ros_node_->advertiseService(get_pos_so);
  }

  ros_queue_thread_ = std::thread([this] { QueueThread(); });
  private_ros_queue_thread_ = std::thread([this] { PrivateQueueThread(); });

  auto update = [this](common::UpdateInfo const & /*info*/) { OnUpdate(); };
  this->update_connection_ = event::Events::ConnectWorldUpdateBegin(update);
}

bool Position3dPlugin::OnRegister(peter_msgs::RegisterPosition3DControllerRequest &req,
                                  peter_msgs::RegisterPosition3DControllerResponse &res) {
  auto const link = GetLink(PLUGIN_NAME, world_, req.scoped_link_name);
  if (!link) {
    return true;
  }

  if (req.controller_type == "pid") {
    controllers_map_[req.scoped_link_name] = std::make_unique<LinkPosition3dPIDController>(
        PLUGIN_NAME, link, req.position_only, req.kp_pos, req.kp_vel, req.max_force, req.max_vel, true);
    ROS_DEBUG_STREAM_NAMED(PLUGIN_NAME, "registered PID controller for link " << req.scoped_link_name);
  } else if (req.controller_type == "kinematic") {
    controllers_map_[req.scoped_link_name] =
        std::make_unique<LinkPosition3dKinematicController>(PLUGIN_NAME, link, req.position_only, req.fixed_rot);
    ROS_DEBUG_STREAM_NAMED(PLUGIN_NAME, "registered kinematic controller for link " << req.scoped_link_name);
  } else {
    ROS_ERROR_STREAM_NAMED(PLUGIN_NAME, "unimplemented controller type " << req.controller_type);
  }

  return true;
}

bool Position3dPlugin::OnUnregister(peter_msgs::UnregisterPosition3DControllerRequest &req,
                                    peter_msgs::UnregisterPosition3DControllerResponse &res) {
  auto const num_removed = controllers_map_.erase(req.scoped_link_name);
  res.success = num_removed > 0;
  return true;
}

bool Position3dPlugin::OnStop(peter_msgs::Position3DStopRequest &req, peter_msgs::Position3DStopResponse &res) {
  (void)res;  // unused

  auto const it = controllers_map_.find(req.scoped_link_name);
  if (it != controllers_map_.cend()) {
    it->second->OnStop();
  } else {
    ROS_WARN_STREAM_THROTTLE_NAMED(1, PLUGIN_NAME, "No link " << req.scoped_link_name);
  }
  return true;
}

bool Position3dPlugin::OnEnable(peter_msgs::Position3DEnableRequest &req, peter_msgs::Position3DEnableResponse &res) {
  (void)res;

  auto const it = controllers_map_.find(req.scoped_link_name);
  if (it != controllers_map_.cend()) {
    it->second->OnEnable(req.enable);
  } else {
    ROS_WARN_STREAM_THROTTLE_NAMED(1, PLUGIN_NAME, "No link " << req.scoped_link_name);
  }
  return true;
}

bool Position3dPlugin::OnFollow(peter_msgs::Position3DFollowRequest &req, peter_msgs::Position3DFollowResponse &res) {
  (void)res;
  auto const it = controllers_map_.find(req.scoped_link_name);
  if (it != controllers_map_.cend()) {
    it->second->OnFollow(req.frame_id);
  } else {
    ROS_WARN_STREAM_THROTTLE_NAMED(1, PLUGIN_NAME, "No link " << req.scoped_link_name);
  }
  return true;
}

bool Position3dPlugin::OnSetPose(peter_msgs::Pose3DActionRequest &req, peter_msgs::Pose3DActionResponse &res) {
  (void)res;
  auto const it = controllers_map_.find(req.scoped_link_name);
  if (it != controllers_map_.cend()) {
    it->second->OnEnable(true);
    it->second->SetPose(req);
  } else {
    ROS_WARN_STREAM_THROTTLE_NAMED(1, PLUGIN_NAME, "No link " << req.scoped_link_name);
  }
  return true;
}

bool Position3dPlugin::OnSet(peter_msgs::Position3DActionRequest &req, peter_msgs::Position3DActionResponse &res) {
  (void)res;
  auto const it = controllers_map_.find(req.scoped_link_name);
  if (it != controllers_map_.cend()) {
    it->second->OnEnable(true);
    it->second->Set(req);
  } else {
    ROS_WARN_STREAM_THROTTLE_NAMED(1, PLUGIN_NAME, "No link " << req.scoped_link_name);
  }
  return true;
}

bool Position3dPlugin::OnList(peter_msgs::Position3DListRequest &req, peter_msgs::Position3DListResponse &res) {
  (void)req;
  for (auto const &[name, controller] : controllers_map_) {
    res.controller_names.push_back(name);
    res.controller_types.push_back(controller->type);
  }
  return true;
}

bool Position3dPlugin::OnWait(peter_msgs::Position3DWaitRequest &req, peter_msgs::Position3DWaitResponse &res) {
  (void)res;

  // Block, waiting until set point is reached or timeout occurs
  auto const t0 = ros::Time::now();

  for (auto const &scoped_link_name : req.scoped_link_names) {
    auto const it = controllers_map_.find(scoped_link_name);
    if (it != controllers_map_.cend()) {
      auto const &controller = it->second;
      Wait(controller, req.timeout_s, req.tolerance_m, req.tolerance_rad, t0);
    }
  }

  return true;
}

bool Position3dPlugin::OnMove(peter_msgs::Position3DActionRequest &req, peter_msgs::Position3DActionResponse &res) {
  (void)res;
  auto const t0 = ros::Time::now();
  auto const it = controllers_map_.find(req.scoped_link_name);
  if (it != controllers_map_.cend()) {
    auto const &controller = it->second;
    controller->OnEnable(true);
    controller->Set(req);

    // Block, waiting until set point is reached or timeout occurs
    Wait(controller, req.timeout_s, req.tolerance_m, 999.0, t0);
  } else {
    ROS_WARN_STREAM_THROTTLE_NAMED(1, PLUGIN_NAME, "No link " << req.scoped_link_name);
  }

  return true;
}

void Position3dPlugin::Wait(std::unique_ptr<BaseLinkPositionController> const &controller, float const timeout_s,
                            double tolerance_m, double tolerance_rad, ros::Time const &t0) {
  // Block, waiting until set point is reached or timeout occurs
  //  auto const t0 = ros::Time::now();
  while (true) {
    auto const dt = ros::Time::now() - t0;
    auto const pose = controller->Get();
    if (not pose) {
      continue;
    }
    auto const pos = pose->Pos();
    auto const rot = pose->Rot();
    auto const error_m = controller->setpoint_.Pos().Distance(pos);
    auto const max_error_rad = quat_diff(controller->setpoint_.Rot(), rot);
    tolerance_m = std::fmax(tolerance_m, 1e-4);
    tolerance_rad = std::fmax(tolerance_rad, 0.01);
    ROS_DEBUG_STREAM_NAMED(PLUGIN_NAME, "Controller " << controller->scoped_link_name_ << " error_m " << error_m
                                                      << " tolerance_m " << tolerance_m << " max_error_rad "
                                                      << max_error_rad << " tolerance_rad " << tolerance_rad << " dt "
                                                      << dt);
    auto const timeout = dt.toSec() >= timeout_s;
    auto const reached = [&]() {
      if (controller->position_only_) {
        return error_m < tolerance_m;
      } else {
        return error_m < tolerance_m and max_error_rad < tolerance_rad;
      }
    }();
    if (timeout or reached) {
      break;
    }
    ros::Duration(0.01).sleep();
  }
}

bool Position3dPlugin::GetPos(peter_msgs::GetPosition3DRequest &req, peter_msgs::GetPosition3DResponse &res) {
  res.success = false;
  auto const it = controllers_map_.find(req.scoped_link_name);
  if (it != controllers_map_.cend()) {
    auto const pos = it->second->Get();
    if (pos) {
      res.pos = ign_vector_3d_to_point(pos->Pos());
      res.success = true;
      res.enabled = it->second->enabled_;
    }
  } else {
    ROS_WARN_STREAM_THROTTLE_NAMED(1, PLUGIN_NAME, "No link " << req.scoped_link_name);
  }
  return true;
}

void Position3dPlugin::OnUpdate() {
  for (auto &[k, v] : controllers_map_) {
    v->OnUpdate();
  }
}

void Position3dPlugin::QueueThread() {
  double constexpr timeout = 0.01;
  while (ros_node_.ok()) {
    queue_.callAvailable(ros::WallDuration(timeout));
  }
}

void Position3dPlugin::PrivateQueueThread() {
  double constexpr timeout = 0.01;
  while (private_ros_node_->ok()) {
    private_queue_.callAvailable(ros::WallDuration(timeout));
  }
}

}  // namespace gazebo
