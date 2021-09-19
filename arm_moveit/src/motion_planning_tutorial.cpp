#include <ros/ros.h>

#include <moveit/kinematic_constraints/utils.h>
#include <moveit/planning_interface/planning_interface.h>
#include <moveit/planning_pipeline/planning_pipeline.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/conversions.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <moveit_visual_tools/moveit_visual_tools.h>

int main(int argc, char** argv) {
  ros::init(argc, argv, "move_group_tutorial");
  ros::AsyncSpinner spinner(1);
  spinner.start();
  ros::NodeHandle nh;

  std::string const group_name = "both_arms";

  auto robot_model_loader = std::make_shared<robot_model_loader::RobotModelLoader>("hdt_michigan/robot_description");
  auto psm = std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(robot_model_loader);

  std::string const robot_namespace = "hdt_michigan";
  auto const scene_topic = ros::names::append(robot_namespace, "move_group/monitored_planning_scene");
  psm->startSceneMonitor(scene_topic);

  auto robot_model = robot_model_loader->getModel();

  auto robot_state =
      std::make_shared<moveit::core::RobotState>(planning_scene_monitor::LockedPlanningSceneRO(psm)->getCurrentState());

  auto planning_pipeline = std::make_shared<planning_pipeline::PlanningPipeline>(robot_model, nh);

  auto display_publisher =
      nh.advertise<moveit_msgs::DisplayTrajectory>("hdt_michigan/move_group/display_planned_path", 10);
  moveit_msgs::DisplayTrajectory display_trajectory;

  namespace rvt = rviz_visual_tools;
  moveit_visual_tools::MoveItVisualTools visual_tools("robot_root");
  visual_tools.deleteAllMarkers();

  // Pose Goal
  planning_interface::MotionPlanRequest req;
  planning_interface::MotionPlanResponse res;
  geometry_msgs::PoseStamped pose;
  pose.header.frame_id = "robot_root";
  pose.pose.position.x = -0.2;
  pose.pose.position.y = 0.6;
  pose.pose.position.z = 0.4;

  tf2::Quaternion myQuaternion;
  myQuaternion.setRPY(-M_PI_2, 0, 0);
  myQuaternion = myQuaternion.normalize();
  pose.pose.orientation.x = myQuaternion.x();
  pose.pose.orientation.y = myQuaternion.y();
  pose.pose.orientation.z = myQuaternion.z();
  pose.pose.orientation.w = myQuaternion.w();

  std::vector<double> tolerance_pose(3, 0.01);
  std::vector<double> tolerance_angle(3, 0.01);

  req.group_name = group_name;
  auto pose_goal = kinematic_constraints::constructGoalConstraints("left_tool", pose, tolerance_pose, tolerance_angle);
  req.goal_constraints.push_back(pose_goal);

  {
    planning_scene_monitor::LockedPlanningSceneRO lscene(psm);
    planning_pipeline->generatePlan(lscene, req, res);
  }

  if (res.error_code_.val != res.error_code_.SUCCESS) {
    ROS_ERROR("Could not compute plan successfully");
    return 0;
  }

  std::cin.get();

  return 0;
}
