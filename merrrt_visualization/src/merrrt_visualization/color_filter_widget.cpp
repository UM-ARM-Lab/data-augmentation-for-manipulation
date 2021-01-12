#include <iostream>
#include <string>

#include <merrrt_visualization/color_filter_widget.h>
#include <arc_utilities/ros_helpers.hpp>

auto constexpr default_cdcpd_node_name = "cdcpd_node";

namespace merrrt_visualization
{
ColorFilterWidget::ColorFilterWidget(QWidget *parent)
    : rviz::Panel(parent)
{
  ui.setupUi(this);
  connect(ui.min_hue_slider, &QSlider::sliderMoved, this, &ColorFilterWidget::MinHueMoved);
  connect(ui.max_hue_slider, &QSlider::sliderMoved, this, &ColorFilterWidget::MaxHueMoved);
  connect(ui.min_sat_slider, &QSlider::sliderMoved, this, &ColorFilterWidget::MinSatMoved);
  connect(ui.max_sat_slider, &QSlider::sliderMoved, this, &ColorFilterWidget::MaxSatMoved);
  connect(ui.min_val_slider, &QSlider::sliderMoved, this, &ColorFilterWidget::MinValMoved);
  connect(ui.max_val_slider, &QSlider::sliderMoved, this, &ColorFilterWidget::MaxValMoved);
}

void ColorFilterWidget::MinHueMoved(int const position)
{
  auto const cdcpd_node_name = ROSHelpers::GetParamDebugLog<std::string>(nh, "cdcpd_node_name", default_cdcpd_node_name);
  auto const param_name = ros::names::append(cdcpd_node_name, "hue_min");
  auto const value = static_cast<double>(position);
  ros::param::set(param_name, value);
  ui.min_hue_label->setText(QString::number(value));
}

void ColorFilterWidget::MinSatMoved(int const position)
{
  auto const cdcpd_node_name = ROSHelpers::GetParamDebugLog<std::string>(nh, "cdcpd_node_name", default_cdcpd_node_name);
  auto const param_name = ros::names::append(cdcpd_node_name, "saturation_min");
  auto const value = static_cast<double>(position) / 100.0;
  ros::param::set(param_name, value);
  ui.min_sat_label->setText(QString::number(value));
}

void ColorFilterWidget::MinValMoved(int const position)
{
  auto const cdcpd_node_name = ROSHelpers::GetParamDebugLog<std::string>(nh, "cdcpd_node_name", default_cdcpd_node_name);
  auto const param_name = ros::names::append(cdcpd_node_name, "value_min");
  auto const value = static_cast<double>(position) / 100.0;
  ros::param::set(param_name, value);
  ui.min_val_label->setText(QString::number(value));
}

void ColorFilterWidget::MaxHueMoved(int const position)
{
  auto const cdcpd_node_name = ROSHelpers::GetParamDebugLog<std::string>(nh, "cdcpd_node_name", default_cdcpd_node_name);
  auto const param_name = ros::names::append(cdcpd_node_name, "hue_max");
  auto const value = static_cast<double>(position);
  ros::param::set(param_name, value);
  ui.max_hue_label->setText(QString::number(value));
}

void ColorFilterWidget::MaxSatMoved(int const position)
{
  auto const cdcpd_node_name = ROSHelpers::GetParamDebugLog<std::string>(nh, "cdcpd_node_name", default_cdcpd_node_name);
  auto const param_name = ros::names::append(cdcpd_node_name, "saturation_max");
  auto const value = static_cast<double>(position) / 100.0;
  ros::param::set(param_name, value);
  ui.max_sat_label->setText(QString::number(value));
}

void ColorFilterWidget::MaxValMoved(int const position)
{
  auto const cdcpd_node_name = ROSHelpers::GetParamDebugLog<std::string>(nh, "cdcpd_node_name", default_cdcpd_node_name);
  auto const param_name = ros::names::append(cdcpd_node_name, "value_max");
  auto const value = static_cast<double>(position) / 100.0;
  ros::param::set(param_name, value);
  ui.max_val_label->setText(QString::number(value));
}


void ColorFilterWidget::load(const rviz::Config &config)
{
  rviz::Panel::load(config);
}

void ColorFilterWidget::save(rviz::Config config) const
{
  rviz::Panel::save(config);
}

}  // namespace merrrt_visualization

#include <pluginlib/class_list_macros.h>

PLUGINLIB_EXPORT_CLASS(merrrt_visualization::ColorFilterWidget, rviz::Panel)
