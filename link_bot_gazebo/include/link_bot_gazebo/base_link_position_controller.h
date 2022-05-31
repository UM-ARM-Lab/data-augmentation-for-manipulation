#pragma once

#include <optional>

#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <peter_msgs/Position3DActionRequest.h>
#include <peter_msgs/Pose3DActionRequest.h>

#include <geometry_msgs/Point.h>
#include <gazebo/physics/World.hh>
#include <gazebo/physics/Link.hh>

namespace gazebo
{

class BaseLinkPositionController
{
 public:
  BaseLinkPositionController(char const *plugin_name, physics::LinkPtr link, std::string const type, bool position_only,
                             bool fixed_rot);

  void OnUpdate();

  virtual void Update(ignition::math::Pose3d const &setpoint) = 0;

  virtual void OnStop();

  virtual void OnEnable(bool enable);

  void OnFollow(std::string const &frame_id);

  void SetPose(peter_msgs::Pose3DActionRequest action);

  void Set(peter_msgs::Position3DActionRequest action);

  [[nodiscard]] virtual std::optional<ignition::math::Pose3d> Get() const;

  char const *plugin_name_;
  physics::LinkPtr const link_;
  std::string const scoped_link_name_;
  std::string following_frame_id_;
  bool enabled_ = false;
  bool following_ = false;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  ignition::math::Pose3d setpoint_;
  bool position_only_;
  bool fixed_rot_;
  double timeout_s_ = 0;
  double speed_mps_ = 0;
  double speed_rps_ = 0;

  const std::string type;
};

}
