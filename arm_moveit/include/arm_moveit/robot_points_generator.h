#pragma once

#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <ros/ros.h>

#include <eigen3/Eigen/Dense>
#include <string>
#include <vector>

class RobotPointsGenerator {
 public:
  RobotPointsGenerator(double res);

  std::vector<std::string> getLinkModelNames();

  std::vector<Eigen::Vector3d> checkCollision(std::string link_name);

  std::vector<Eigen::Vector3d> pointsToCheck(robot_state::RobotState state, std::string link_name) const;

  std::string getRobotName() const;
 private:
  robot_model_loader::RobotModelLoaderPtr model_loader_;
  robot_model::RobotModelPtr model_;
  planning_scene::PlanningScene scene_;
  collision_detection::WorldPtr world_;
  double res_;
  moveit_visual_tools::MoveItVisualTools visual_tools_;
  ros::Publisher points_to_check_pub_;
  ros::Publisher bbox_pub_;
  ros::NodeHandle nh_;
  std::shared_ptr<shapes::Sphere> sphere_shape_;
};
