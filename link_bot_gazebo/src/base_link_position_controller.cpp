#include <link_bot_gazebo/base_link_position_controller.h>
#include <link_bot_gazebo/gazebo_plugin_utils.h>
#include <ros/console.h>

namespace gazebo {

BaseLinkPositionController::BaseLinkPositionController(char const *plugin_name, physics::LinkPtr link,
                                                       std::string const type, bool position_only, bool fixed_rot)
    : plugin_name_(plugin_name),
      link_(link),
      scoped_link_name_(link->GetScopedName()),
      tf_listener_(tf_buffer_),
      setpoint_(link->WorldPose()),
      position_only_(position_only),
      fixed_rot_(fixed_rot),
      type(type) {}

std::optional<ignition::math::Pose3d> BaseLinkPositionController::Get() const {
  if (!link_) {
    return {};
  }
  return link_->WorldPose();
}

void BaseLinkPositionController::OnFollow(std::string const &frame_id) {
  OnEnable(true);
  following_ = true;
  following_frame_id_ = frame_id;
  speed_mps_ = std::numeric_limits<typeof(speed_mps_)>::max();
  speed_rps_ = std::numeric_limits<typeof(speed_rps_)>::max();
}

void BaseLinkPositionController::OnUpdate() {
  if (not enabled_) {
    return;
  }

  if (following_) {
    try {
      auto const transform_stamped = tf_buffer_.lookupTransform("world", following_frame_id_, ros::Time(0));
      auto const trans = transform_stamped.transform.translation;
      auto const rot = transform_stamped.transform.rotation;
      Update({trans.x, trans.y, trans.z, rot.w, rot.x, rot.y, rot.z});
    } catch (tf2::TransformException &ex) {
      ROS_WARN_STREAM_NAMED(plugin_name_, ex.what());
    }
  } else {
    Update(setpoint_);
  }
}

void BaseLinkPositionController::SetPose(peter_msgs::Pose3DActionRequest action) {
  OnEnable(true);
  following_ = false;
  setpoint_ = pose_to_ign_pose(action.pose);
  timeout_s_ = action.timeout_s;
  speed_mps_ = action.speed_mps;
  speed_rps_ = action.speed_rps;
}

void BaseLinkPositionController::Set(peter_msgs::Position3DActionRequest action) {
  OnEnable(true);
  following_ = false;
  setpoint_.Pos() = (point_to_ign_vector_3d(action.position));
  timeout_s_ = action.timeout_s;
  speed_mps_ = action.speed_mps;
}

void BaseLinkPositionController::OnStop() {
  auto const pos = Get();
  if (pos) {
    setpoint_ = *pos;
  }
}

void BaseLinkPositionController::OnEnable(bool enable) {
  enabled_ = enable;
  OnStop();
}

}  // namespace gazebo
