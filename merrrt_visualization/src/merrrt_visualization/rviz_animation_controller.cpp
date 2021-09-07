#include <merrrt_visualization/rviz_animation_controller.h>
#include <peter_msgs/AnimationControl.h>
#include <peter_msgs/GetAnimControllerState.h>
#include <std_msgs/Empty.h>

#include <QApplication>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QPushButton>
#include <QSettings>
#include <iostream>

#define create_service_options(type, name, bind) \
  ros::AdvertiseServiceOptions::create<type>(name, bind, ros::VoidPtr(), &queue_)

namespace merrrt_visualization {
RVizAnimationController::RVizAnimationController(QWidget *parent) : rviz::Panel(parent) {
  ui.setupUi(this);
  connect(ui.forward_button, &QPushButton::clicked, this, &RVizAnimationController::ForwardClicked);
  connect(ui.backward_button, &QPushButton::clicked, this, &RVizAnimationController::BackwardClicked);
  connect(ui.play_forward_button, &QPushButton::clicked, this, &RVizAnimationController::PlayForwardClicked);
  connect(ui.play_backward_button, &QPushButton::clicked, this, &RVizAnimationController::PlayBackwardClicked);
  connect(ui.pause_button, &QPushButton::clicked, this, &RVizAnimationController::PauseClicked);
  connect(ui.done_button, &QPushButton::clicked, this, &RVizAnimationController::DoneClicked);
  connect(ui.loop_checkbox, &QCheckBox::toggled, this, &RVizAnimationController::LoopToggled);
  connect(ui.auto_play_checkbox, &QCheckBox::toggled, this, &RVizAnimationController::AutoPlayToggled);
  connect(ui.period_spinbox, qOverload<double>(&QDoubleSpinBox::valueChanged), this,
          &RVizAnimationController::PeriodChanged);
  connect(ui.step_number_lineedit, &QLineEdit::returnPressed, this, &RVizAnimationController::StepNumberChanged);
  connect(this, &RVizAnimationController::setStepText, ui.step_number_lineedit, &QLineEdit::setText,
          Qt::QueuedConnection);
  connect(this, &RVizAnimationController::setMaxText, ui.max_step_number_label, &QLabel::setText, Qt::QueuedConnection);

  command_pub_ = ros_node_.advertise<peter_msgs::AnimationControl>("rviz_anim/control", 10);

  auto get_state_bind = [this](peter_msgs::GetAnimControllerStateRequest &req,
                               peter_msgs::GetAnimControllerStateResponse &res) {
    (void)req;  // unused
    res.state.auto_play = ui.auto_play_checkbox->isChecked();
    res.state.loop = ui.loop_checkbox->isChecked();
    res.state.period = static_cast<float>(ui.period_spinbox->value());
    return true;
  };
  auto get_state_so = create_service_options(peter_msgs::GetAnimControllerState, "rviz_anim/get_state", get_state_bind);
  get_state_srv_ = ros_node_.advertiseService(get_state_so);

  // this is stupid why must I list this type here but not when I do this for services!?
  boost::function<void(const std_msgs::Int64::ConstPtr &)> time_cb = [this](const std_msgs::Int64::ConstPtr &msg) {
    TimeCallback(msg);
  };
  auto time_sub_so = ros::SubscribeOptions::create("rviz_anim/time", 10, time_cb, ros::VoidPtr(), &queue_);
  time_sub_ = ros_node_.subscribe(time_sub_so);

  // this is stupid why must I list this type here but not when I do this for services!?
  boost::function<void(const std_msgs::Int64::ConstPtr &)> max_time_cb = [this](const std_msgs::Int64::ConstPtr &msg) {
    MaxTimeCallback(msg);
  };
  auto max_time_sub_so = ros::SubscribeOptions::create("rviz_anim/max_time", 10, max_time_cb, ros::VoidPtr(), &queue_);
  max_time_sub_ = ros_node_.subscribe(max_time_sub_so);

  ros_queue_thread_ = std::thread([this] { QueueThread(); });
}

RVizAnimationController::~RVizAnimationController() {
  queue_.clear();
  queue_.disable();
  ros_node_.shutdown();
  ros_queue_thread_.join();
}

void RVizAnimationController::TimeCallback(const std_msgs::Int64::ConstPtr &msg) {
  {
    const QSignalBlocker blocker(ui.step_number_lineedit);
    emit setStepText(QString::number(msg->data));
  }
  update();
}

void RVizAnimationController::MaxTimeCallback(const std_msgs::Int64::ConstPtr &msg) {
  auto const text = QString::number(msg->data);
  emit setMaxText(text);
  update();
}

void RVizAnimationController::DoneClicked() {
  peter_msgs::AnimationControl cmd;
  cmd.command = peter_msgs::AnimationControl::DONE;
  command_pub_.publish(cmd);
}

void RVizAnimationController::ForwardClicked() {
  peter_msgs::AnimationControl cmd;
  cmd.command = peter_msgs::AnimationControl::STEP_FORWARD;
  command_pub_.publish(cmd);
}

void RVizAnimationController::BackwardClicked() {
  peter_msgs::AnimationControl cmd;
  cmd.command = peter_msgs::AnimationControl::STEP_BACKWARD;
  command_pub_.publish(cmd);
}

void RVizAnimationController::PauseClicked() {
  peter_msgs::AnimationControl cmd;
  cmd.command = peter_msgs::AnimationControl::PAUSE;
  command_pub_.publish(cmd);
}

void RVizAnimationController::PlayForwardClicked() {
  peter_msgs::AnimationControl cmd;
  cmd.command = peter_msgs::AnimationControl::PLAY_FORWARD;
  command_pub_.publish(cmd);
}

void RVizAnimationController::PlayBackwardClicked() {
  peter_msgs::AnimationControl cmd;
  cmd.command = peter_msgs::AnimationControl::PLAY_BACKWARD;
  command_pub_.publish(cmd);
}

void RVizAnimationController::LoopToggled() {
  peter_msgs::AnimationControl cmd;
  cmd.state.loop = ui.loop_checkbox->isChecked();
  cmd.command = peter_msgs::AnimationControl::SET_LOOP;
  command_pub_.publish(cmd);

  if (ui.auto_play_checkbox->isChecked()) {
    ROS_WARN_STREAM("Auto-play takes precedence over looping");
    // show a warning here
  }
}

void RVizAnimationController::AutoPlayToggled() {
  peter_msgs::AnimationControl cmd;
  cmd.state.auto_play = ui.auto_play_checkbox->isChecked();
  cmd.command = peter_msgs::AnimationControl::SET_AUTO_PLAY;
  command_pub_.publish(cmd);

  if (ui.loop_checkbox->isChecked()) {
    ROS_WARN_STREAM("Looping takes precedence over auto-play");
    // show a warning here
  }
}

void RVizAnimationController::StepNumberChanged() {
  auto const idx = ui.step_number_lineedit->text().toInt();
  peter_msgs::AnimationControl cmd;
  cmd.state.idx = idx;
  cmd.command = peter_msgs::AnimationControl::SET_IDX;
  command_pub_.publish(cmd);
}

void RVizAnimationController::PeriodChanged(double period) {
  peter_msgs::AnimationControl cmd;
  cmd.state.period = period;
  cmd.command = peter_msgs::AnimationControl::SET_PERIOD;
  command_pub_.publish(cmd);
}

void RVizAnimationController::QueueThread() {
  double constexpr timeout = 0.01;
  while (ros_node_.ok()) {
    queue_.callAvailable(ros::WallDuration(timeout));
  }
}

void RVizAnimationController::load(const rviz::Config &config) {
  rviz::Panel::load(config);

  bool loop;
  if (config.mapGetBool("loop", &loop)) {
    ui.loop_checkbox->setChecked(loop);
  }

  float period;
  if (config.mapGetFloat("period", &period)) {
    ui.period_spinbox->setValue(period);
  }

  bool auto_play;
  if (config.mapGetBool("auto_play", &auto_play)) {
    ui.auto_play_checkbox->setChecked(auto_play);
  }
}

void RVizAnimationController::save(rviz::Config config) const {
  rviz::Panel::save(config);
  config.mapSetValue("auto_play", ui.auto_play_checkbox->isChecked());
  config.mapSetValue("loop", ui.loop_checkbox->isChecked());
  config.mapSetValue("period", ui.period_spinbox->value());
}

}  // namespace merrrt_visualization

#include <pluginlib/class_list_macros.h>

PLUGINLIB_EXPORT_CLASS(merrrt_visualization::RVizAnimationController, rviz::Panel)