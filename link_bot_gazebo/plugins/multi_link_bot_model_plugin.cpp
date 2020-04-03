#include "multi_link_bot_model_plugin.h"

#include <std_srvs/EmptyRequest.h>

#include <cstdio>
#include <gazebo/common/Time.hh>
#include <gazebo/common/Timer.hh>
#include <ignition/math/Vector3.hh>
#include <memory>
#include <sstream>

namespace gazebo {

constexpr auto close_enough{0.001};

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-vararg"
void MultiLinkBotModelPlugin::Load(physics::ModelPtr const parent, sdf::ElementPtr const sdf)
{
  // Make sure the ROS node for Gazebo has already been initialized
  if (!ros::isInitialized()) {
    auto argc = 0;
    char **argv = nullptr;
    ros::init(argc, argv, "multi_link_bot_model_plugin", ros::init_options::NoSigintHandler);
  }

  auto joy_bind = boost::bind(&MultiLinkBotModelPlugin::OnJoy, this, _1);
  auto joy_so = ros::SubscribeOptions::create<sensor_msgs::Joy>("joy", 1, joy_bind, ros::VoidPtr(), &queue_);
  auto execute_abs_action_bind = boost::bind(&MultiLinkBotModelPlugin::ExecuteAbsoluteAction, this, _1, _2);
  auto execute_abs_action_so = ros::AdvertiseServiceOptions::create<peter_msgs::ExecuteAction>(
      "execute_absolute_action", execute_abs_action_bind, ros::VoidPtr(), &queue_);
  auto action_bind = boost::bind(&MultiLinkBotModelPlugin::ExecuteAction, this, _1, _2);
  auto action_so = ros::AdvertiseServiceOptions::create<peter_msgs::ExecuteAction>("execute_action", action_bind,
                                                                                   ros::VoidPtr(), &queue_);
  auto action_mode_bind = boost::bind(&MultiLinkBotModelPlugin::OnActionMode, this, _1);
  auto action_mode_so = ros::SubscribeOptions::create<std_msgs::String>("link_bot_action_mode", 1, action_mode_bind,
                                                                        ros::VoidPtr(), &queue_);
  auto state_bind = boost::bind(&MultiLinkBotModelPlugin::StateServiceCallback, this, _1, _2);
  auto service_so = ros::AdvertiseServiceOptions::create<peter_msgs::LinkBotState>("link_bot_state", state_bind,
                                                                                   ros::VoidPtr(), &queue_);
  constexpr auto gripper_service_name{"gripper"};
  auto get_object_gripper_bind = boost::bind(&MultiLinkBotModelPlugin::GetObjectGripperCallback, this, _1, _2);
  auto get_object_gripper_so = ros::AdvertiseServiceOptions::create<peter_msgs::GetObject>(
      gripper_service_name, get_object_gripper_bind, ros::VoidPtr(), &queue_);
  constexpr auto link_bot_service_name{"link_bot"};
  auto get_object_link_bot_bind = boost::bind(&MultiLinkBotModelPlugin::GetObjectLinkBotCallback, this, _1, _2);
  auto get_object_link_bot_so = ros::AdvertiseServiceOptions::create<peter_msgs::GetObject>(
      link_bot_service_name, get_object_link_bot_bind, ros::VoidPtr(), &queue_);
  auto reset_bind = boost::bind(&MultiLinkBotModelPlugin::ResetRobot, this, _1, _2);
  auto reset_so = ros::AdvertiseServiceOptions::create<peter_msgs::LinkBotReset>("reset_robot", reset_bind,
                                                                                 ros::VoidPtr(), &queue_);

  joy_sub_ = ros_node_.subscribe(joy_so);
  execute_action_service_ = ros_node_.advertiseService(action_so);
  execute_absolute_action_service_ = ros_node_.advertiseService(execute_abs_action_so);
  register_object_pub_ = ros_node_.advertise<std_msgs::String>("register_object", 10, true);
  reset_service_ = ros_node_.advertiseService(reset_so);
  action_mode_sub_ = ros_node_.subscribe(action_mode_so);
  state_service_ = ros_node_.advertiseService(service_so);
  get_object_gripper_service_ = ros_node_.advertiseService(get_object_gripper_so);
  get_object_link_bot_service_ = ros_node_.advertiseService(get_object_link_bot_so);
  objects_service_ = ros_node_.serviceClient<peter_msgs::GetObjects>("objects");

  ros_queue_thread_ = std::thread(std::bind(&MultiLinkBotModelPlugin::QueueThread, this));
  execute_trajs_ros_queue_thread_ = std::thread(std::bind(&MultiLinkBotModelPlugin::QueueThread, this));

  while (register_object_pub_.getNumSubscribers() < 1) {
  }

  {
    std_msgs::String register_object;
    register_object.data = link_bot_service_name;
    register_object_pub_.publish(register_object);
  }

  {
    std_msgs::String register_object;
    register_object.data = gripper_service_name;
    register_object_pub_.publish(register_object);
  }

  model_ = parent;

  {
    if (!sdf->HasElement("rope_length")) {
      printf("using default rope length=%f\n", length_);
    }
    else {
      length_ = sdf->GetElement("rope_length")->Get<double>();
    }

    if (!sdf->HasElement("num_links")) {
      printf("using default num_links=%u\n", num_links_);
    }
    else {
      num_links_ = sdf->GetElement("num_links")->Get<unsigned int>();
    }

    if (!sdf->HasElement("kP_pos")) {
      printf("using default kP_pos=%f\n", kP_pos_);
    }
    else {
      kP_pos_ = sdf->GetElement("kP_pos")->Get<double>();
    }

    if (!sdf->HasElement("kI_pos")) {
      printf("using default kI_pos=%f\n", kI_pos_);
    }
    else {
      kI_pos_ = sdf->GetElement("kI_pos")->Get<double>();
    }

    if (!sdf->HasElement("kD_pos")) {
      printf("using default kD_pos=%f\n", kD_pos_);
    }
    else {
      kD_pos_ = sdf->GetElement("kD_pos")->Get<double>();
    }

    if (!sdf->HasElement("kP_vel")) {
      printf("using default kP_vel=%f\n", kP_vel_);
    }
    else {
      kP_vel_ = sdf->GetElement("kP_vel")->Get<double>();
    }

    if (!sdf->HasElement("kI_vel")) {
      printf("using default kI_vel=%f\n", kI_vel_);
    }
    else {
      kI_vel_ = sdf->GetElement("kI_vel")->Get<double>();
    }

    if (!sdf->HasElement("kD_vel")) {
      printf("using default kD_vel=%f\n", kD_vel_);
    }
    else {
      kD_vel_ = sdf->GetElement("kD_vel")->Get<double>();
    }

    if (!sdf->HasElement("gripper1_link")) {
      throw std::invalid_argument("no gripper1_link tag provided");
    }

    if (!sdf->HasElement("max_force")) {
      printf("using default max_force=%f\n", max_force_);
    }
    else {
      max_force_ = sdf->GetElement("max_force")->Get<double>();
    }

    if (!sdf->HasElement("max_vel")) {
      printf("using default max_vel=%f\n", max_vel_);
    }
    else {
      max_vel_ = sdf->GetElement("max_vel")->Get<double>();
    }

    if (!sdf->HasElement("A")) {
      printf("identity A matrix");
    }
    else {
      auto const a_vec = sdf->GetElement("A")->Get<ignition::math::Vector4d>();
      A_(0, 0) = a_vec[0];
      A_(0, 1) = a_vec[1];
      A_(1, 0) = a_vec[2];
      A_(1, 1) = a_vec[3];
      gzlog << "Using A Matrix: " << A_ << '\n';
    }

    if (!sdf->HasElement("B")) {
      printf("identity B matrix");
    }
    else {
      auto const b_vec = sdf->GetElement("B")->Get<ignition::math::Vector4d>();
      B_(0, 0) = b_vec[0];
      B_(0, 1) = b_vec[1];
      B_(1, 0) = b_vec[2];
      B_(1, 1) = b_vec[3];
      gzlog << "Using B Matrix: " << B_ << '\n';
    }
  }

  ros_node_.setParam("n_action", 2);
  ros_node_.setParam("link_bot/rope_length", length_);
  ros_node_.setParam("max_speed", max_speed_);

  auto const &gripper1_link_name = sdf->GetElement("gripper1_link")->Get<std::string>();
  gripper1_link_ = model_->GetLink(gripper1_link_name);

  updateConnection_ = event::Events::ConnectWorldUpdateBegin(std::bind(&MultiLinkBotModelPlugin::OnUpdate, this));
  constexpr auto max_integral{0};
  gripper1_pos_pid_ = common::PID(kP_pos_, kI_pos_, kD_pos_, max_integral, -max_integral, max_vel_, -max_vel_);

  constexpr auto max_vel_integral{1};
  gripper1_vel_pid_ =
      common::PID(kP_vel_, kI_vel_, kD_vel_, max_vel_integral, -max_vel_integral, max_force_, -max_force_);

  gzlog << "MultiLinkBot Model Plugin finished initializing!\n";
}
#pragma clang diagnostic pop

peter_msgs::NamedPoints MultiLinkBotModelPlugin::GetConfiguration()
{
  peter_msgs::NamedPoints configuration;
  configuration.name = "link_bot";
  for (auto link_idx{1U}; link_idx <= num_links_; ++link_idx) {
    std::stringstream ss;
    ss << "link_" << link_idx;
    auto link_name = ss.str();
    auto const link = model_->GetLink(link_name);
    peter_msgs::NamedPoint named_point;
    named_point.point.x = link->WorldPose().Pos().X();
    named_point.point.y = link->WorldPose().Pos().Y();
    named_point.name = link_name;
    configuration.points.emplace_back(named_point);
  }

  auto const head = model_->GetLink("head");
  peter_msgs::NamedPoint named_point;
  named_point.point.x = head->WorldPose().Pos().X();
  named_point.point.y = head->WorldPose().Pos().Y();
  named_point.name = "head";
  configuration.points.emplace_back(named_point);

  return configuration;
}

auto MultiLinkBotModelPlugin::GetGripper1Pos() -> ignition::math::Vector3d const
{
  auto p = gripper1_link_->WorldPose().Pos();
  // zero the Z component
  p.Z(0);
  return p;
}

auto MultiLinkBotModelPlugin::GetGripper1Vel() -> ignition::math::Vector3d const
{
  auto v = gripper1_link_->WorldLinearVel();
  v.Z(0);
  return v;
}

ControlResult MultiLinkBotModelPlugin::UpdateControl()
{
  std::lock_guard<std::mutex> guard(control_mutex_);
  auto const dt = model_->GetWorld()->Physics()->GetMaxStepSize();
  ControlResult control_result{};

  auto const gripper1_pos = GetGripper1Pos();
  auto const gripper1_vel_ = GetGripper1Vel();

  // Gripper 1
  {
    if (mode_ == "position") {
      gripper1_pos_error_ = gripper1_pos - gripper1_target_position_;
      auto const target_vel = gripper1_pos_pid_.Update(gripper1_pos_error_.Length(), dt);
      auto const gripper1_target_velocity = gripper1_pos_error_.Normalized() * target_vel;

      auto const gripper1_vel_error = gripper1_vel_ - gripper1_target_velocity;
      auto const force_mag = gripper1_vel_pid_.Update(gripper1_vel_error.Length(), dt);
      control_result.gripper1_force = gripper1_vel_error.Normalized() * force_mag;
    }
    else if (mode_ == "disabled") {
      // do nothing!
    }
  }

  return control_result;
}

void MultiLinkBotModelPlugin::OnUpdate()
{
  ControlResult control = UpdateControl();

  gripper1_link_->AddForce(control.gripper1_force);
}

void MultiLinkBotModelPlugin::OnJoy(sensor_msgs::JoyConstPtr const msg)
{
  constexpr auto scale{2000.0 / 32768.0};
  gripper1_target_position_.X(gripper1_target_position_.X() - msg->axes[0] * scale);
  gripper1_target_position_.Y(gripper1_target_position_.Y() + msg->axes[1] * scale);
}

bool MultiLinkBotModelPlugin::ExecuteAbsoluteAction(peter_msgs::ExecuteActionRequest &req,
                                                    peter_msgs::ExecuteActionResponse &res)
{
  mode_ = "position";

  ignition::math::Vector3d position{req.action.action[0], req.action.action[1], 0};
  gripper1_target_position_ = position;

  auto const seconds_per_step = model_->GetWorld()->Physics()->GetMaxStepSize();
  auto const steps = static_cast<unsigned int>(req.action.max_time_per_step / seconds_per_step);
  // Wait until the setpoint is reached
  model_->GetWorld()->Step(steps);

  // TODO: fill out state here
  res.needs_reset = false;

  // stop by setting the current position as the target
  gripper1_target_position_ = GetGripper1Pos();

  return true;
}

bool MultiLinkBotModelPlugin::ExecuteAction(peter_msgs::ExecuteActionRequest &req,
                                            peter_msgs::ExecuteActionResponse &res)
{
  mode_ = "position";

  Eigen::Vector2d s;
  s(0) = gripper1_target_position_.X();
  s(1) = gripper1_target_position_.Y();

  Eigen::Vector2d u;
  u(0) = req.action.action[0];
  u(1) = req.action.action[1];

  Eigen::Vector2d const s_ = A_ * s + B_ * u;
  gripper1_target_position_.X(s_(0));
  gripper1_target_position_.Y(s_(1));

  auto const seconds_per_step = model_->GetWorld()->Physics()->GetMaxStepSize();
  auto const steps = static_cast<unsigned int>(req.action.max_time_per_step / seconds_per_step);
  // Wait until the setpoint is reached
  model_->GetWorld()->Step(steps);

  // TODO: fill out state here
  res.needs_reset = false;

  // stop by setting the current position as the target
  gripper1_target_position_ = GetGripper1Pos();

  return true;
}

void MultiLinkBotModelPlugin::OnActionMode(std_msgs::StringConstPtr msg) { mode_ = msg->data; }

bool MultiLinkBotModelPlugin::StateServiceCallback(peter_msgs::LinkBotStateRequest &req,
                                                   peter_msgs::LinkBotStateResponse &res)
{
  // get all links named "link_%d" where d is in [1, num_links)
  for (auto const &link : model_->GetLinks()) {
    auto const name = link->GetName();
    int link_idx;
    auto const n_matches = sscanf(name.c_str(), "link_%d", &link_idx);
    if (n_matches == 1 and link_idx >= 1 and link_idx <= num_links_) {
      geometry_msgs::Point pt;
      pt.x = link->WorldPose().Pos().X();
      pt.y = link->WorldPose().Pos().Y();
      res.points.emplace_back(pt);
      res.link_names.emplace_back(name);
    }
  }

  auto const link = model_->GetLink("head");
  geometry_msgs::Point pt;
  pt.x = link->WorldPose().Pos().X();
  pt.y = link->WorldPose().Pos().Y();
  res.points.emplace_back(pt);
  res.link_names.emplace_back("head");

  res.gripper1_force.x = gripper1_vel_pid_.GetCmd();
  res.gripper1_force.z = 0;

  auto const gripper1_velocity = gripper1_link_->WorldLinearVel();
  res.gripper1_velocity.x = gripper1_velocity.X();
  res.gripper1_velocity.y = gripper1_velocity.Y();
  res.gripper1_velocity.z = 0;

  res.header.stamp = ros::Time::now();

  return true;
}

bool MultiLinkBotModelPlugin::GetObjectGripperCallback(peter_msgs::GetObjectRequest &req,
                                                       peter_msgs::GetObjectResponse &res)
{
  auto const link = model_->GetLink("head");
  float const x = link->WorldPose().Pos().X();
  float const y = link->WorldPose().Pos().Y();
  peter_msgs::NamedPoint head_point;
  geometry_msgs::Point pt;
  head_point.point.x = x;
  head_point.point.y = y;
  head_point.name = "gripper";
  res.object.points.emplace_back(head_point);
  res.object.state_vector = std::vector<float>{static_cast<float>(x), y};
  res.object.name = "gripper";

  return true;
}

bool MultiLinkBotModelPlugin::GetObjectLinkBotCallback(peter_msgs::GetObjectRequest &req,
                                                       peter_msgs::GetObjectResponse &res)
{
  res.object.name = "link_bot";
  std::vector<float> state_vector;
  for (auto link_idx{1U}; link_idx <= num_links_; ++link_idx) {
    std::stringstream ss;
    ss << "link_" << link_idx;
    auto link_name = ss.str();
    auto const link = model_->GetLink(link_name);
    peter_msgs::NamedPoint named_point;
    float const x = link->WorldPose().Pos().X();
    float const y = link->WorldPose().Pos().Y();
    state_vector.push_back(x);
    state_vector.push_back(y);
    named_point.point.x = x;
    named_point.point.y = y;
    named_point.name = link_name;
    res.object.points.emplace_back(named_point);
  }

  auto const link = model_->GetLink("head");
  peter_msgs::NamedPoint head_point;
  geometry_msgs::Point pt;
  float const x = link->WorldPose().Pos().X();
  float const y = link->WorldPose().Pos().Y();
  state_vector.push_back(x);
  state_vector.push_back(y);
  head_point.point.x = x;
  head_point.point.y = y;
  res.object.state_vector = state_vector;
  head_point.name = "head";
  res.object.points.emplace_back(head_point);

  return true;
}

bool MultiLinkBotModelPlugin::ResetRobot(peter_msgs::LinkBotResetRequest &req, peter_msgs::LinkBotResetResponse &res)
{
  gripper1_target_position_.X(req.point.x);
  gripper1_target_position_.Y(req.point.y);

  while (true) {
    auto const seconds_per_step = model_->GetWorld()->Physics()->GetMaxStepSize();
    auto const steps = static_cast<unsigned int>(1.0 / seconds_per_step);
    // Wait until the setpoint is reached
    model_->GetWorld()->Step(steps);
    if (gripper1_pos_error_.Length() < close_enough) {
      break;
    }
  }

  return true;
}

void MultiLinkBotModelPlugin::QueueThread()
{
  double constexpr timeout = 0.01;
  while (ros_node_.ok()) {
    queue_.callAvailable(ros::WallDuration(timeout));
    execute_trajs_queue_.callAvailable(ros::WallDuration(timeout));
  }
}

MultiLinkBotModelPlugin::~MultiLinkBotModelPlugin()
{
  queue_.clear();
  queue_.disable();
  execute_trajs_queue_.clear();
  execute_trajs_queue_.disable();
  ros_node_.shutdown();
  ros_queue_thread_.join();
  execute_trajs_ros_queue_thread_.join();
}

// Register this plugin with the simulator
GZ_REGISTER_MODEL_PLUGIN(MultiLinkBotModelPlugin)
}  // namespace gazebo
