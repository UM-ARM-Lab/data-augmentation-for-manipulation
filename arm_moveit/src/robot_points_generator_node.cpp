#include <arm_moveit/robot_points_generator.h>

int main(int argc, char **argv) {
  ros::init(argc, argv, "robot_points_generator");
  auto const res = 0.02;
  RobotPointsGenerator robot_points_generator(res);
  auto const links = robot_points_generator.getLinkModelNames();
  std::map<std::string, std::vector<Eigen::Vector3d>> points;
  for (auto const &link_to_check : links) {
    auto const points_for_link = robot_points_generator.checkCollision(link_to_check);
    points.emplace(link_to_check, points_for_link);
  }

  ros::Duration(1).sleep();

  return EXIT_SUCCESS;
}
