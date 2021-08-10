#pragma once

#include <geometry_msgs/Point.h>
#include <peter_msgs/GetPosition3D.h>
#include <peter_msgs/Position3DAction.h>
#include <peter_msgs/Position3DEnable.h>

#include <functional>
#include <gazebo/common/Events.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/common/Time.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/transport/TransportTypes.hh>
#include <link_bot_gazebo/base_link_position_controller.h>

namespace gazebo
{
class LinkPosition3dPIDController : public BaseLinkPositionController
{
 public:
  LinkPosition3dPIDController(char const *plugin_name,
                              physics::LinkPtr link,
                              bool position_only,
                              double kp_pos,
                              double kp_vel,
                              double max_force,
                              double max_vel,
                              bool grav_comp);

  void Update(ignition::math::Pose3d const &setpoint) override;

  double kP_pos_;
  double kP_vel_;
  double max_force_;
  double max_vel_;

  double kD_pos_{0.0};
  double kI_vel_{0.0};
  double kD_vel_{0.0};

  common::PID pos_pid_;
  common::PID vel_pid_;
  ignition::math::Vector3d pos_error_{0, 0, 0};
  double total_mass_{0.0};
  bool gravity_compensation_{false};
  double z_integral_{0.0};
};

}  // namespace gazebo
