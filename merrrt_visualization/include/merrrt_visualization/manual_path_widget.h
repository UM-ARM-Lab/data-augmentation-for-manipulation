#pragma once

#include <ros/ros.h>
#include <rviz/panel.h>
#include <rviz/rviz_export.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>

#include <QObject>
#include <QWidget>

#include "ui_manual_path_widget.h"

namespace merrrt_visualization
{
class ManualPathWidget : public rviz::Panel
{
  Q_OBJECT

public:
  explicit ManualPathWidget(QWidget *parent = nullptr);

  void load(const rviz::Config &config) override;
  void save(rviz::Config config) const override;

public slots:

private:
  Ui_ManualPathWidget ui;

  ros::NodeHandle nh;
};

}  // namespace merrrt_visualization
