#include <iostream>
#include <string>

#include <merrrt_visualization/manual_path_widget.h>
#include <arc_utilities/ros_helpers.hpp>

namespace merrrt_visualization
{
ManualPathWidget::ManualPathWidget(QWidget *parent)
    : rviz::Panel(parent)
{
  ui.setupUi(this);
}


void ManualPathWidget::load(const rviz::Config &config)
{
  rviz::Panel::load(config);
}

void ManualPathWidget::save(rviz::Config config) const
{
  rviz::Panel::save(config);
}

}  // namespace merrrt_visualization

#include <pluginlib/class_list_macros.h>

PLUGINLIB_EXPORT_CLASS(merrrt_visualization::ManualPathWidget, rviz::Panel)
