#include "linkbot_model_plugin.h"

#include <ignition/math/Vector3.hh>
#include <memory>

namespace gazebo {

bool in_contact(msgs::Contacts const &contacts)
{
  for (auto i{0u}; i < contacts.contact_size(); ++i) {
    auto const &contact = contacts.contact(i);
    if (contact.collision1() == "link_bot::head::head_collision" and
        contact.collision2() != "ground_plane::link::collision") {
      return true;
    }
    if (contact.collision2() == "link_bot::head::head_collision" and
        contact.collision1() != "ground_plane::link::collision") {
      return true;
    }
  }
  return false;
}

void LinkBotModelPlugin::Load(physics::ModelPtr const parent, sdf::ElementPtr const sdf)
{
  // Make sure the ROS node for Gazebo has already been initalized
  // Initialize ros, if it has not already bee initialized.
  if (!ros::isInitialized()) {
    auto argc = 0;
    char **argv = nullptr;
    ros::init(argc, argv, "linkbot_model_plugin", ros::init_options::NoSigintHandler);
  }

  ros_node_ = std::make_unique<ros::NodeHandle>("linkbot_model_plugin");

  auto joy_bind = boost::bind(&LinkBotModelPlugin::OnJoy, this, _1);
  auto joy_so = ros::SubscribeOptions::create<sensor_msgs::Joy>("/joy", 1, joy_bind, ros::VoidPtr(), &queue_);
  auto vel_action_bind = boost::bind(&LinkBotModelPlugin::OnVelocityAction, this, _1);
  auto vel_action_so = ros::SubscribeOptions::create<link_bot_gazebo::LinkBotVelocityAction>(
      "/link_bot_velocity_action", 1, vel_action_bind, ros::VoidPtr(), &queue_);
  auto force_action_bind = boost::bind(&LinkBotModelPlugin::OnForceAction, this, _1);
  auto force_action_so = ros::SubscribeOptions::create<link_bot_gazebo::LinkBotForceAction>(
      "/link_bot_force_action", 1, force_action_bind, ros::VoidPtr(), &queue_);
  auto config_bind = boost::bind(&LinkBotModelPlugin::OnConfiguration, this, _1);
  auto config_so = ros::SubscribeOptions::create<link_bot_gazebo::LinkBotConfiguration>(
      "/link_bot_configuration", 1, config_bind, ros::VoidPtr(), &queue_);
  auto state_bind = boost::bind(&LinkBotModelPlugin::StateServiceCallback, this, _1, _2);
  auto service_so = ros::AdvertiseServiceOptions::create<link_bot_gazebo::LinkBotState>("/link_bot_state", state_bind,
                                                                                        ros::VoidPtr(), &queue_);

  joy_sub_ = ros_node_->subscribe(joy_so);
  vel_cmd_sub_ = ros_node_->subscribe(vel_action_so);
  force_cmd_sub_ = ros_node_->subscribe(force_action_so);
  config_sub_ = ros_node_->subscribe(config_so);
  state_service_ = ros_node_->advertiseService(service_so);

  ros_queue_thread_ = std::thread(std::bind(&LinkBotModelPlugin::QueueThread, this));

  if (!sdf->HasElement("kP")) {
    printf("using default kP=%f\n", kP_);
  }
  else {
    kP_ = sdf->GetElement("kP")->Get<double>();
  }

  if (!sdf->HasElement("kI")) {
    printf("using default kI=%f\n", kI_);
  }
  else {
    kI_ = sdf->GetElement("kI")->Get<double>();
  }

  if (!sdf->HasElement("kD")) {
    printf("using default kD=%f\n", kD_);
  }
  else {
    kD_ = sdf->GetElement("kD")->Get<double>();
  }

  if (!sdf->HasElement("action_scale")) {
    printf("using default action_scale=%f\n", action_scale);
  }
  else {
    action_scale = sdf->GetElement("action_scale")->Get<double>();
  }

  ROS_INFO("kP=%f, kI=%f, kD=%f", kP_, kI_, kD_);

  model_ = parent;

  updateConnection_ = event::Events::ConnectWorldUpdateBegin(std::bind(&LinkBotModelPlugin::OnUpdate, this));
  x_vel_pid_ = common::PID(kP_, kI_, kD_, 100, -100, 800, -800);
  y_vel_pid_ = common::PID(kP_, kI_, kD_, 100, -100, 800, -800);

  for (auto const &sensor : sensors::SensorManager::Instance()->GetSensors()) {
    if (sensor->Type() == "contact") {
      auto const contact_sensor = std::dynamic_pointer_cast<sensors::ContactSensor>(sensor);
      contact_sensors_.emplace_back(contact_sensor);
    }
  }
}

void LinkBotModelPlugin::OnUpdate()
{
  ignition::math::Vector3d force{};
  if (use_force_) {
    auto i{0u};
    auto const &links = model_->GetLinks();
    for (auto &link : links) {
      force.X(wrenches_[i].force.x);
      force.Y(wrenches_[i].force.y);
      link->AddForce(force);
      ++i;
    }
  }
  else {
    if (velocity_control_link_) {
      auto const current_linear_vel = velocity_control_link_->WorldLinearVel();
      auto const error = current_linear_vel - target_linear_vel_;
      force.X(x_vel_pid_.Update(error.X(), 0.001));
      force.Y(y_vel_pid_.Update(error.Y(), 0.001));
      velocity_control_link_->AddForce(force);
    }
  }
}

void LinkBotModelPlugin::OnJoy(sensor_msgs::JoyConstPtr const msg)
{
  use_force_ = false;
  velocity_control_link_ = model_->GetLink("head");
  if (not velocity_control_link_) {
    std::cout << "invalid link pointer. Link name "
              << "head"
              << " is not one of:\n";
    for (auto const &link : model_->GetLinks()) {
      std::cout << link->GetName() << "\n";
    }
    return;
  }
  target_linear_vel_.X(-msg->axes[0] * action_scale);
  target_linear_vel_.Y(msg->axes[1] * action_scale);
}

void LinkBotModelPlugin::OnVelocityAction(link_bot_gazebo::LinkBotVelocityActionConstPtr const msg)
{
  use_force_ = false;
  velocity_control_link_ = model_->GetLink(msg->control_link_name);
  if (not velocity_control_link_) {
    std::cout << "invalid link pointer. Link name " << msg->control_link_name << " is not one of:\n";
    for (auto const &link : model_->GetLinks()) {
      std::cout << link->GetName() << "\n";
    }
    return;
  }
  target_linear_vel_.X(msg->vx * action_scale);
  target_linear_vel_.Y(msg->vy * action_scale);
}

void LinkBotModelPlugin::OnForceAction(link_bot_gazebo::LinkBotForceActionConstPtr const msg)
{
  use_force_ = true;
  auto const &joints = model_->GetJoints();
  if (msg->wrenches.size() != joints.size()) {
    ROS_ERROR("Model as %lu joints config message had %lu", joints.size(), msg->wrenches.size());
    return;
  }

  wrenches_ = msg->wrenches;
}

void LinkBotModelPlugin::OnConfiguration(link_bot_gazebo::LinkBotConfigurationConstPtr msg)
{
  auto const &joints = model_->GetJoints();

  if (joints.size() != msg->joint_angles_rad.size()) {
    ROS_ERROR("Model as %lu joints config message had %lu", joints.size(), msg->joint_angles_rad.size());
    return;
  }

  ignition::math::Pose3d pose{};
  pose.Pos().X(msg->tail_pose.x);
  pose.Pos().Y(msg->tail_pose.y);
  pose.Pos().Z(0.05);
  pose.Rot() = ignition::math::Quaterniond::EulerToQuaternion(0, 0, msg->tail_pose.theta);
  model_->SetWorldPose(pose);
  model_->SetWorldTwist({0, 0, 0}, {0, 0, 0});

  for (size_t i = 0; i < joints.size(); ++i) {
    auto const &joint = joints[i];
    joint->SetPosition(0, msg->joint_angles_rad[i]);
    joint->SetVelocity(0, 0);
  }
}

bool LinkBotModelPlugin::StateServiceCallback(link_bot_gazebo::LinkBotStateRequest &req,
                                              link_bot_gazebo::LinkBotStateResponse &res)
{
  auto const &tail = model_->GetLink("link_0");
  auto const &mid = model_->GetLink("link_4");
  auto const &head = model_->GetLink("head");

  for (auto const contact_sensor : contact_sensors_) {
    auto const &contacts = contact_sensor->Contacts();
    res.in_contact.emplace_back(in_contact(contacts));
  }
  auto const &tail_torque = tail->RelativeTorque();
  auto const &mid_torque = mid->RelativeTorque();
  auto const &head_torque = head->RelativeTorque();

  res.tail_x = tail->WorldPose().Pos().X();
  res.tail_y = tail->WorldPose().Pos().Y();
  res.mid_x = mid->WorldPose().Pos().X();
  res.mid_y = mid->WorldPose().Pos().Y();
  res.head_x = head->WorldPose().Pos().X();
  res.head_y = head->WorldPose().Pos().Y();
  res.tail_torque.x = tail_torque.X();
  res.tail_torque.y = tail_torque.Y();
  res.tail_torque.z = tail_torque.Z();
  res.mid_torque.x = mid_torque.X();
  res.mid_torque.y = mid_torque.Y();
  res.mid_torque.z = mid_torque.Z();
  res.head_torque.x = head_torque.X();
  res.head_torque.y = head_torque.Y();
  res.head_torque.z = head_torque.Z();
  res.overstretched = 0;
  return true;
}

void LinkBotModelPlugin::QueueThread()
{
  double constexpr timeout = 0.01;
  while (ros_node_->ok()) {
    queue_.callAvailable(ros::WallDuration(timeout));
  }
}

LinkBotModelPlugin::~LinkBotModelPlugin()
{
  queue_.clear();
  queue_.disable();
  ros_node_->shutdown();
  ros_queue_thread_.join();
}

// Register this plugin with the simulator
GZ_REGISTER_MODEL_PLUGIN(LinkBotModelPlugin)
}  // namespace gazebo
