#include <link_bot_gazebo/link_position_3d_kinematic_controller.h>
#include <ros/console.h>

#include <gazebo/physics/physics.hh>
#include <ignition/math/Quaternion.hh>
#include <link_bot_gazebo/mymath.hpp>

namespace gazebo {

LinkPosition3dKinematicController::LinkPosition3dKinematicController(char const *plugin_name, physics::LinkPtr link,
                                                                     bool position_only, bool fixed_rot)
    : BaseLinkPositionController(plugin_name, link, "kinematic", position_only, fixed_rot) {}

void LinkPosition3dKinematicController::Update(ignition::math::Pose3d const &setpoint) {
  if (!link_) {
    ROS_ERROR_STREAM_NAMED(plugin_name_, "pointer to the link " << scoped_link_name_ << " is null");
    return;
  }

  // compute the current output position
  auto const current_pose = Get();
  if (not current_pose) {
    ROS_ERROR_STREAM_NAMED(plugin_name_, "failed to get link " << scoped_link_name_ << " state");
    return;
  }
  auto const current_position = current_pose->Pos();
  auto const current_rot = current_pose->Rot();

  // update the output position, this is what makes "speed" mean something
  auto const dt = link_->GetWorld()->Physics()->GetMaxStepSize();
  // step_size is a decimal, from 0 to 1. take a step from current to setpoint
  auto const max_step_size_m = 0.005;
  auto const max_step_size_rad = 0.01;
  auto const delta_distance = std::fmin(speed_mps_ * dt, max_step_size_m);
  auto const delta_rad = std::fmin(speed_rps_ * dt, max_step_size_rad);
  auto const distance = current_position.Distance(setpoint.Pos());
  auto const distance_rot = quat_diff(current_rot, setpoint.Rot());
  //  ROS_DEBUG_STREAM_NAMED(plugin_name_, scoped_link_name_ << " distance_rot " << distance_rot << " delta_rad " <<
  //  delta_rad); ROS_DEBUG_STREAM_NAMED(plugin_name_, scoped_link_name_ << " distance " << distance << " delta_distance
  //  " << delta_distance);
  auto const step_size = std::fmin(delta_distance / distance, 1);
  auto const step_size_rad = std::fmin(delta_rad / distance_rot, 1);
  auto const direction = (setpoint.Pos() - current_position);
  auto const direction_rot = (setpoint.Rot() - current_rot);
  auto const output_position = [&]() {
    if (distance > 1e-4) {
      return current_position + direction * step_size;
    } else {
      return setpoint.Pos();
    }
  }();
  auto const output_orientation = [&]() {
    if (fixed_rot_) {
      return ignition::math::Quaterniond(1, 0, 0, 0);
    } else if (position_only_) {
      return current_rot;
    } else if (distance_rot < 0.01) {
      return setpoint.Rot();
    } else {
      return current_rot + direction_rot * step_size_rad;
    }
  }();

  if (!output_position.IsFinite()) {
    ROS_ERROR_STREAM_NAMED(plugin_name_, ""
                                             << " current " << current_position.X() << " " << current_position.Y()
                                             << " " << current_position.Z() << " setpoint " << setpoint_.X() << " "
                                             << setpoint_.Y() << " " << setpoint_.Z() << " distance " << distance
                                             << " dt " << dt << " step " << step_size << " output "
                                             << output_position.X() << " " << output_position.Y() << " "
                                             << output_position.Z());
    return;
  }

  // actually move the link
  link_->SetAngularVel(ignition::math::Vector3d::Zero);
  link_->SetLinearVel(ignition::math::Vector3d::Zero);
  ignition::math::Pose3d pose{output_position, output_orientation};
  ROS_DEBUG_STREAM_THROTTLE_NAMED(1.0, plugin_name_, "Setting pose of " << scoped_link_name_ << " to " << pose);
  link_->SetWorldPose(pose);
}

void LinkPosition3dKinematicController::OnEnable(bool enable) {
  BaseLinkPositionController::OnEnable(enable);
  while (true) {
    // I'm seeing that calling "SetKinematic" actually just toggles kinematic,
    // so this ensure it gets set to the right value
    link_->SetKinematic(enable);
    auto const is_k = link_->GetKinematic();
    ROS_DEBUG_STREAM_NAMED(plugin_name_, "enable =" << enable << ". is kinematic = " << link_->GetKinematic());
    if (is_k == enable) {
      break;
    }
  }
}

}  // namespace gazebo