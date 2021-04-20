#include <merrrt_visualization/merrrt_widget.h>

#include <iostream>

auto constexpr LOGNAME = "MerrrtWidget";

FilterWidget::FilterWidget(QWidget *parent, std::string const &name) : QWidget(parent) {
  ui.setupUi(this);
  ui.label->setText(QString::fromStdString(name));
}

int FilterWidget::GetFilterType() {
  // order in the UI file is assumed to match the constants in the .msg file
  return ui.combobox->currentIndex();
}

namespace merrrt_visualization {
MerrrtWidget::MerrrtWidget(QWidget *parent)
    : rviz::Panel(parent), world_control_srv_(ros_node_.serviceClient<peter_msgs::WorldControl>("world_control")) {
  ui.setupUi(this);
  label_sub_ = ros_node_.subscribe<peter_msgs::LabelStatus>("label_viz", 10, &MerrrtWidget::LabelCallback, this);
  error_sub_ = ros_node_.subscribe<std_msgs::Float32>("error", 10, &MerrrtWidget::ErrorCallback, this);
  stdev_sub_ = ros_node_.subscribe<std_msgs::Float32>("stdev", 10, &MerrrtWidget::StdevCallback, this);
  accept_probability_sub_ =
      ros_node_.subscribe<std_msgs::Float32>("accept_probability_viz", 10, &MerrrtWidget::OnAcceptProbability, this);
  traj_idx_sub_ = ros_node_.subscribe<std_msgs::Float32>("traj_idx_viz", 10, &MerrrtWidget::OnTrajIdx, this);
  recov_prob_sub_ = ros_node_.subscribe<std_msgs::Float32>("recovery_probability_viz", 10,
                                                           &MerrrtWidget::OnRecoveryProbability, this);

  viz_options_srv_ = ros_node_.advertiseService("get_viz_options", &MerrrtWidget::GetVizOptions, this);
}

void MerrrtWidget::OnTrajIdx(const std_msgs::Float32::ConstPtr &msg) {
  auto const text = QString::asprintf("%0.4f", msg->data);
  ui.traj_idx->setText(text);
}

void MerrrtWidget::ErrorCallback(const std_msgs::Float32::ConstPtr &msg) {
  auto const text = QString::asprintf("%0.4f", msg->data);
  ui.error->setText(text);
}

void MerrrtWidget::StdevCallback(const std_msgs::Float32::ConstPtr &msg) {
  auto const text = QString::asprintf("%0.4f", msg->data);
  ui.stdev->setText(text);
}

void MerrrtWidget::LabelCallback(const peter_msgs::LabelStatus::ConstPtr &msg) {
  if (msg->status == peter_msgs::LabelStatus::Accept) {
    ui.bool_indicator->setStyleSheet("background-color: rgb(0, 200, 0);");
  } else if (msg->status == peter_msgs::LabelStatus::Reject) {
    ui.bool_indicator->setStyleSheet("background-color: rgb(200, 0, 0);");
  } else {
    ui.bool_indicator->setStyleSheet("background-color: rgb(150, 150, 150);");
  }
}

void MerrrtWidget::OnRecoveryProbability(const std_msgs::Float32::ConstPtr &msg) {
  auto const blue = 50;
  auto red = 0;
  auto green = 0;
  if (msg->data >= 0 and msg->data <= 1) {
    // *0.8 to cool the colors
    auto const cool_factor = 0.7;
    red = static_cast<int>(255 * (1 - msg->data) * cool_factor);
    green = static_cast<int>(255 * msg->data * cool_factor);
  } else {
    red = 0;
    green = 0;
  }
  ui.recovery_probability->setStyleSheet(QString("color: rgb(%1, %2, %3);").arg(red).arg(green).arg(blue));
  auto const text = QString::asprintf("%0.4f", msg->data);
  ui.recovery_probability->setText(text);
}

void MerrrtWidget::OnAcceptProbability(const std_msgs::Float32::ConstPtr &msg) {
  auto const blue = 50;
  auto red = 0;
  auto green = 0;
  if (msg->data >= 0 and msg->data <= 1) {
    // *0.8 to cool the colors
    auto const cool_factor = 0.7;
    red = static_cast<int>(255 * (1 - msg->data) * cool_factor);
    green = static_cast<int>(255 * msg->data * cool_factor);
  } else {
    red = 0;
    green = 0;
  }
  ui.accept_probability->setStyleSheet(QString("color: rgb(%1, %2, %3);").arg(red).arg(green).arg(blue));
  auto const text = QString::asprintf("%0.4f", msg->data);
  ui.accept_probability->setText(text);
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
