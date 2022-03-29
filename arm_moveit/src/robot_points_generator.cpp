#include <arm_moveit/robot_points_generator.h>
#include <jsk_recognition_msgs/BoundingBox.h>

std::string const collision_sphere_name = "collision_sphere";
constexpr auto const LOGGER_NAME = "robot_points_generator";

RobotPointsGenerator::RobotPointsGenerator(double const res, std::string const &robot_description, double collision_sphere_radius)
    : model_loader_(std::make_shared<robot_model_loader::RobotModelLoader>(robot_description)),
      model_(model_loader_->getModel()),
      scene_(model_),
      res_(res),
      collision_sphere_radius_(collision_sphere_radius),
      collision_sphere_diameter_(2 * collision_sphere_radius),
      visual_tools_("robot_root", "/moveit_visual_markers", model_),
      sphere_shape_(std::make_shared<shapes::Sphere>(collision_sphere_radius)) {
  world_ = scene_.getWorldNonConst();

  points_to_check_pub_ = nh_.advertise<visualization_msgs::Marker>("points_to_check", 10, true);
  bbox_pub_ = nh_.advertise<jsk_recognition_msgs::BoundingBox>("link_bbox", 10, true);

  Eigen::Isometry3d initial_pose{Eigen::Isometry3d::Identity()};
  world_->addToObject(collision_sphere_name, sphere_shape_, initial_pose);
}

std::vector<std::string> RobotPointsGenerator::getLinkModelNames() { return model_->getLinkModelNames(); }

std::vector<Eigen::Vector3d> RobotPointsGenerator::checkCollision(std::string const &link_name,
                                                                  std::string const &frame_id) {
  std::vector<Eigen::Vector3d> points_frame_id;
  std::vector<Eigen::Vector3d> debug_viz_points;

  // make a copy of the ACM
  auto acm = scene_.getAllowedCollisionMatrix();
  // initial ignore (allow) collision between every link and the collision sphere
  for (auto const &l : getLinkModelNames()) {
    acm.setEntry(l, collision_sphere_name, true);
  }
  // not explicitly set collision to be NOT allowed (not ignored) for the one link we're checking
  acm.setEntry(link_name, collision_sphere_name, false);

  auto const state = scene_.getCurrentState();

  auto const &frame_id_transform = state.getGlobalLinkTransform(frame_id);

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
  viz_point_to_check_msg.scale.x = collision_sphere_diameter_;
  viz_point_to_check_msg.scale.y = collision_sphere_diameter_;
  viz_point_to_check_msg.scale.z = collision_sphere_diameter_;
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
      Eigen::Vector3d point_frame_id = frame_id_transform.inverse() * point_robot_frame;
      points_frame_id.push_back(point_frame_id);
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
  viz_occupied_points_msg.scale.x = collision_sphere_diameter_;
  viz_occupied_points_msg.scale.y = collision_sphere_diameter_;
  viz_occupied_points_msg.scale.z = collision_sphere_diameter_;
  viz_occupied_points_msg.pose.orientation.w = 1;
  for (auto const &point : debug_viz_points) {
    geometry_msgs::Point viz_point;
    viz_point.x = point.x();
    viz_point.y = point.y();
    viz_point.z = point.z();
    viz_occupied_points_msg.points.push_back(viz_point);
  }

  points_to_check_pub_.publish(viz_occupied_points_msg);

  return points_frame_id;
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cert-flp30-c"
std::vector<Eigen::Vector3d> RobotPointsGenerator::pointsToCheck(robot_state::RobotState const &state,
                                                                 std::string const &link_name) const {
  std::vector<Eigen::Vector3d> points_to_check;

  // look up the position of the link in robot frame
  // given a max side length (meters) to check, figure out how many points we need to check
  auto const link = state.getLinkModel(link_name);
  Eigen::Vector3d const &collision_bbox_shape_link_frame = link->getShapeExtentsAtOrigin();

  if (collision_bbox_shape_link_frame.norm() == 0) {
    return points_to_check;
  }

  Eigen::Vector3d const offset = link->getCenteredBoundingBoxOffset();

  Eigen::Vector3d res_vec = Eigen::Vector3d::Ones() * res_ * 2;
  Eigen::Vector3d const lower_link_frame = -collision_bbox_shape_link_frame / 2 - res_vec + offset;
  Eigen::Vector3d const upper_link_frame = collision_bbox_shape_link_frame / 2 + res_vec + offset;

  auto const &link_to_robot_root_transform = state.getGlobalLinkTransform(link_name);

  jsk_recognition_msgs::BoundingBox bbox;
  bbox.header.frame_id = link_name;
  bbox.pose.position.x = offset.x();
  bbox.pose.position.y = offset.y();
  bbox.pose.position.z = offset.z();
  bbox.pose.orientation.w = 1;
  bbox.dimensions.x = collision_bbox_shape_link_frame.x();
  bbox.dimensions.y = collision_bbox_shape_link_frame.y();
  bbox.dimensions.z = collision_bbox_shape_link_frame.z();
  bbox_pub_.publish(bbox);

  for (auto p_i_x = lower_link_frame.x(); p_i_x <= upper_link_frame.x(); p_i_x += res_) {
    for (auto p_i_y = lower_link_frame.y(); p_i_y <= upper_link_frame.y(); p_i_y += res_) {
      for (auto p_i_z = lower_link_frame.z(); p_i_z <= upper_link_frame.z(); p_i_z += res_) {
        Eigen::Vector3d const p_i_link_frame(p_i_x, p_i_y, p_i_z);
        Eigen::Vector3d const &p_i_robot_frame = link_to_robot_root_transform * p_i_link_frame.homogeneous();
        points_to_check.emplace_back(p_i_robot_frame);
      }
    }
  }

  return points_to_check;
}
std::string RobotPointsGenerator::getRobotName() const { return model_->getName(); }
#pragma clang diagnostic pop
