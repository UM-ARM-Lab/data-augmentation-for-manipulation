#include <arm_moveit/robot_points_generator.h>

std::string const collision_sphere_name = "collision_sphere";
constexpr double const collision_sphere_radius = 0.005;
constexpr double const collision_sphere_diameter = collision_sphere_radius * 2;
constexpr auto const LOGGER_NAME = "robot_points_generator";

RobotPointsGenerator::RobotPointsGenerator(double const res)
    : model_loader_(std::make_shared<robot_model_loader::RobotModelLoader>()),
      model_(model_loader_->getModel()),
      scene_(model_),
      res_(res),
      visual_tools_("robot_root", "/moveit_visual_markers", model_),
      sphere_shape_(std::make_shared<shapes::Sphere>(collision_sphere_radius)) {
  world_ = scene_.getWorldNonConst();

  points_to_check_pub_ = nh_.advertise<visualization_msgs::Marker>("points_to_check", 10, true);

  Eigen::Isometry3d initial_pose{Eigen::Isometry3d::Identity()};
  world_->addToObject(collision_sphere_name, sphere_shape_, initial_pose);
}

std::vector<std::string> RobotPointsGenerator::getLinkModelNames() { return model_->getLinkModelNames(); }

std::vector<Eigen::Vector3d> RobotPointsGenerator::checkCollision(std::string link_name) {
  std::vector<Eigen::Vector3d> points_link_frame;
  std::vector<Eigen::Vector3d> debug_viz_points;

  // make a copy of the ACM
  auto acm = scene_.getAllowedCollisionMatrix();
  // initial ignore (allow) collision between every link and the collision sphere
  for (auto const &l : getLinkModelNames()) {
    acm.setEntry(l, collision_sphere_name, true);
  }
  // not explicitly set collision to be NOT allowed (not ignored) for the one link we're checking
  acm.setEntry(link_name, collision_sphere_name, false);
  acm.print(std::cout);

  auto const state = scene_.getCurrentState();

  auto const link_transform = state.getGlobalLinkTransform(link_name);

  visual_tools_.publishRobotState(state);

  auto request = collision_detection::CollisionRequest();
  request.contacts = true;
  request.distance = false;
  request.max_contacts = 1;
  request.max_contacts_per_pair = 1;
  request.cost = false;
  request.verbose = false;
  auto result = collision_detection::CollisionResult();

  visualization_msgs::Marker delete_msg;
  delete_msg.ns = "point_to_check";
  delete_msg.action = visualization_msgs::Marker::DELETEALL;
  points_to_check_pub_.publish(delete_msg);

  visualization_msgs::Marker viz_point_to_check_msg;
  viz_point_to_check_msg.header.frame_id = "robot_root";
  viz_point_to_check_msg.action = visualization_msgs::Marker::ADD;
  viz_point_to_check_msg.ns = "point_to_check";
  viz_point_to_check_msg.id = 0;
  viz_point_to_check_msg.type = visualization_msgs::Marker::SPHERE_LIST;
  viz_point_to_check_msg.scale.x = collision_sphere_diameter;
  viz_point_to_check_msg.scale.y = collision_sphere_diameter;
  viz_point_to_check_msg.scale.z = collision_sphere_diameter;
  viz_point_to_check_msg.pose.orientation.w = 1;

  for (auto const &point_robot_frame : pointsToCheck(state, link_name)) {
    Eigen::Isometry3d collision_sphere_pose{Eigen::Isometry3d::Identity()};
    collision_sphere_pose.translate(point_robot_frame);
    world_->clearObjects();
    world_->addToObject(collision_sphere_name, sphere_shape_, collision_sphere_pose);

    result.clear();
    scene_.checkCollision(request, result, state, acm);

    geometry_msgs::Point viz_point;
    viz_point.x = point_robot_frame.x();
    viz_point.y = point_robot_frame.y();
    viz_point.z = point_robot_frame.z();
    viz_point_to_check_msg.points.push_back(viz_point);
    std_msgs::ColorRGBA viz_point_color;
    viz_point_color.a = 1;
    if (result.collision) {
      debug_viz_points.push_back(point_robot_frame);
      Eigen::Vector3d point_link_frame = link_transform * point_robot_frame;
      points_link_frame.push_back(point_link_frame);
      viz_point_color.r = 1;
    } else {
      viz_point_color.g = 1;
    }
    viz_point_to_check_msg.colors.push_back(viz_point_color);
  }

  points_to_check_pub_.publish(viz_point_to_check_msg);

  visualization_msgs::Marker viz_occupied_points_msg;
  viz_occupied_points_msg.header.frame_id = "robot_root";
  viz_occupied_points_msg.action = visualization_msgs::Marker::ADD;
  viz_occupied_points_msg.ns = "occupied_points";
  viz_occupied_points_msg.id = 0;
  viz_occupied_points_msg.type = visualization_msgs::Marker::SPHERE_LIST;
  viz_occupied_points_msg.color.b = 1;
  viz_occupied_points_msg.color.a = 1;
  viz_occupied_points_msg.scale.x = collision_sphere_diameter;
  viz_occupied_points_msg.scale.y = collision_sphere_diameter;
  viz_occupied_points_msg.scale.z = collision_sphere_diameter;
  viz_occupied_points_msg.pose.orientation.w = 1;
  for (auto const &point : debug_viz_points) {
    geometry_msgs::Point viz_point;
    viz_point.x = point.x();
    viz_point.y = point.y();
    viz_point.z = point.z();
    viz_occupied_points_msg.points.push_back(viz_point);
  }

  points_to_check_pub_.publish(viz_occupied_points_msg);

  return points_link_frame;
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cert-flp30-c"
std::vector<Eigen::Vector3d> RobotPointsGenerator::pointsToCheck(robot_state::RobotState state,
                                                                 std::string link_name) const {
  std::vector<Eigen::Vector3d> points_to_check;

  // look up the position of the link in robot frame
  // given a max side length (meters) to check, figure out how many points we need to check
  auto const link = state.getLinkModel(link_name);
  Eigen::Vector3d const &collision_bbox_shape_link_frame = link->getShapeExtentsAtOrigin();

  if (collision_bbox_shape_link_frame.norm() == 0) {
    return points_to_check;
  }

  auto const link_transform = state.getGlobalLinkTransform(link_name);
  Eigen::Vector3d collision_bbox_shape = (link_transform.rotation() * collision_bbox_shape_link_frame).cwiseAbs();
  auto const link_origin = link_transform.translation();
  Eigen::Vector3d res_vec = Eigen::Vector3d::Ones() * res_ * 2;
  Eigen::Vector3d lower = link_origin - collision_bbox_shape / 2 - res_vec;
  Eigen::Vector3d upper = link_origin + collision_bbox_shape / 2 + res_vec;

  auto const &offset_link_frame = link->getCenteredBoundingBoxOffset();
  Eigen::Vector3d offset = link_transform.rotation() * offset_link_frame;

  for (auto p_i_x = lower.x(); p_i_x <= upper.x(); p_i_x += res_) {
    for (auto p_i_y = lower.y(); p_i_y <= upper.y(); p_i_y += res_) {
      for (auto p_i_z = lower.z(); p_i_z <= upper.z(); p_i_z += res_) {
        Eigen::Vector3d p_i(p_i_x, p_i_y, p_i_z);
        Eigen::Vector3d p_i_robot_frame = p_i + offset;
        points_to_check.emplace_back(p_i_robot_frame);
      }
    }
  }

  return points_to_check;
}
#pragma clang diagnostic pop

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
