#pragma once

#include <ros/ros.h>
#include <rviz/panel.h>
#include <rviz/rviz_export.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>

#include <QObject>
#include <QWidget>

#include "ui_color_filter_widget.h"

namespace merrrt_visualization
{
class ColorFilterWidget : public rviz::Panel
{
  Q_OBJECT

public:
  explicit ColorFilterWidget(QWidget *parent = nullptr);

  void load(const rviz::Config &config) override;
  void save(rviz::Config config) const override;

public slots:
  void MinHueMoved(int position);
  void MinSatMoved(int position);
  void MinValMoved(int position);

  void MaxHueMoved(int position);
  void MaxSatMoved(int position);
  void MaxValMoved(int position);

private:
  Ui_ColorFilterWidget ui;

  ros::NodeHandle nh;
};

}  // namespace merrrt_visualization
