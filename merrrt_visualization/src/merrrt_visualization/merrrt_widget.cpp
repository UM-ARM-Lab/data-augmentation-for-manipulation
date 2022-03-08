#include <merrrt_visualization/merrrt_widget.h>

#include <iostream>

auto constexpr LOGNAME = "MerrrtWidget";

FilterWidget::FilterWidget(QWidget *parent, std::string const &name) : QWidget(parent) {
  ui.setupUi(this);
  ui.label->setText(QString::fromStdString(name));
}

int FilterWidget::GetFilterType() const {
  // order in the UI file is assumed to match the constants in the .msg file
  return ui.combobox->currentIndex();
}

namespace merrrt_visualization {
MerrrtWidget::MerrrtWidget(QWidget *parent)
    : rviz::Panel(parent), world_control_srv_(ros_node_.serviceClient<peter_msgs::WorldControl>("world_control")) {
  ui.setupUi(this);

  connect(this, &MerrrtWidget::setAcceptProbText, ui.accept_probability, &QLabel::setText);
  connect(this, &MerrrtWidget::setRecoveryProbText, ui.recovery_probability, &QLabel::setText);
  connect(this, &MerrrtWidget::setErrorText, ui.error, &QLabel::setText);
  connect(this, &MerrrtWidget::setPredErrorText, ui.pred_error, &QLabel::setText);
  connect(this, &MerrrtWidget::setStdevText, ui.stdev, &QLabel::setText);
  connect(this, &MerrrtWidget::setTrajIdxText, ui.traj_idx, &QLabel::setText);
  connect(this, &MerrrtWidget::setWeightText, ui.weight, &QLabel::setText);

  label_sub_ = ros_node_.subscribe<peter_msgs::LabelStatus>("label_viz", 10, &MerrrtWidget::LabelCallback, this);
  error_sub_ = ros_node_.subscribe<std_msgs::Float32>("error", 10, &MerrrtWidget::ErrorCallback, this);
  pred_error_sub_ = ros_node_.subscribe<std_msgs::Float32>("pred_error_viz", 10, &MerrrtWidget::PredErrorCallback, this);
  stdev_sub_ = ros_node_.subscribe<std_msgs::Float32>("stdev", 10, &MerrrtWidget::StdevCallback, this);
  accept_probability_sub_ =
      ros_node_.subscribe<std_msgs::Float32>("accept_probability_viz", 10, &MerrrtWidget::OnAcceptProbability, this);
  traj_idx_sub_ = ros_node_.subscribe<std_msgs::Float32>("traj_idx_viz", 10, &MerrrtWidget::OnTrajIdx, this);
  weight_sub_ = ros_node_.subscribe<std_msgs::Float32>("weight_viz", 10, &MerrrtWidget::OnWeight, this);
  recov_prob_sub_ = ros_node_.subscribe<std_msgs::Float32>("recovery_probability_viz", 10,
                                                           &MerrrtWidget::OnRecoveryProbability, this);
  viz_options_srv_ = ros_node_.advertiseService("get_viz_options", &MerrrtWidget::GetVizOptions, this);
}

void MerrrtWidget::OnWeight(const std_msgs::Float32::ConstPtr &msg) {
  ui.weight->setStyleSheet(redGreenTextColor(msg->data));
  auto const text = QString::asprintf("%0.4f", msg->data);
  emit setWeightText(text);
}

void MerrrtWidget::OnTrajIdx(const std_msgs::Float32::ConstPtr &msg) {
  auto const text = QString::asprintf("%0.4f", msg->data);
  emit setTrajIdxText(text);
}

void MerrrtWidget::PredErrorCallback(const std_msgs::Float32::ConstPtr &msg) {
  auto const text = QString::asprintf("%0.4f", msg->data);
  emit setPredErrorText(text);
}

void MerrrtWidget::ErrorCallback(const std_msgs::Float32::ConstPtr &msg) {
  auto const text = QString::asprintf("%0.4f", msg->data);
  emit setErrorText(text);
  ui.error->setText(text);
}

void MerrrtWidget::StdevCallback(const std_msgs::Float32::ConstPtr &msg) {
  auto const text = QString::asprintf("%0.4f", msg->data);
  emit setStdevText(text);
  ui.stdev->setText(text);
}

void MerrrtWidget::LabelCallback(const peter_msgs::LabelStatus::ConstPtr &msg) const {
  if (msg->status == peter_msgs::LabelStatus::Accept) {
    ui.bool_indicator->setStyleSheet("background-color: rgb(0, 200, 0);");
  } else if (msg->status == peter_msgs::LabelStatus::Reject) {
    ui.bool_indicator->setStyleSheet("background-color: rgb(200, 0, 0);");
  } else {
    ui.bool_indicator->setStyleSheet("background-color: rgb(150, 150, 150);");
  }
}

void MerrrtWidget::OnRecoveryProbability(const std_msgs::Float32::ConstPtr &msg) {
  ui.recovery_probability->setStyleSheet(redGreenTextColor(msg->data));
  auto const text = QString::asprintf("%0.4f", msg->data);
  emit setRecoveryProbText(text);
  ui.recovery_probability->setText(text);
}

void MerrrtWidget::OnAcceptProbability(const std_msgs::Float32::ConstPtr &msg) {
  ui.accept_probability->setStyleSheet(redGreenTextColor(msg->data));
  auto const text = QString::asprintf("%0.4f", msg->data);
  emit setAcceptProbText(text);
  ui.accept_probability->setText(text);
}
QString MerrrtWidget::redGreenTextColor(float const x) {
  auto const blue = 50;
  int red;
  int green;
  if (x >= 0 and x <= 1) {
    // *0.8 to cool the colors
    auto const cool_factor = 0.7;
    red = static_cast<int>(255 * (1 - x) * cool_factor);
    green = static_cast<int>(255 * x * cool_factor);
  } else {
    red = 0;
    green = 0;
  }

  return QString("color: rgb(%1, %2, %3);").arg(red).arg(green).arg(blue);
}

bool MerrrtWidget::GetVizOptions(peter_msgs::GetVizOptions::Request &req, peter_msgs::GetVizOptions::Response &res) {
  res.viz_options.accumulate = ui.accumulate_checkbox->isChecked();
  for (auto const &name : req.names) {
    // if a widget for name doe not exist, create the widget and store it
    if (filter_widgets.find(name) == filter_widgets.end()) {
      ROS_DEBUG_STREAM_NAMED(LOGNAME, "Adding filter for " << name);
      // create a new three-way picker defining the filter
      auto *new_widget = new FilterWidget(this, name);
      ui.filter_layout->addWidget(new_widget);
      filter_widgets[name] = new_widget;
    }
  }
  for (auto [name, filter_widget] : filter_widgets) {
    auto const filter_type = filter_widget->GetFilterType();
    ROS_DEBUG_STREAM_NAMED(LOGNAME, "Get state " << filter_type << " for filter " << name);
    res.viz_options.names.emplace_back(name);
    res.viz_options.filter_types.emplace_back(filter_type);
  }
  return true;
}

void MerrrtWidget::load(const rviz::Config &config) { rviz::Panel::load(config); }

void MerrrtWidget::save(rviz::Config config) const { rviz::Panel::save(config); }

}  // namespace merrrt_visualization

#include <pluginlib/class_list_macros.h>

PLUGINLIB_EXPORT_CLASS(merrrt_visualization::MerrrtWidget, rviz::Panel)
