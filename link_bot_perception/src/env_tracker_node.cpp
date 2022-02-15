#include <message_filters/subscriber.h>
//#include <octomap_ros
#include <moveit_msgs/PlanningScene.h>
#include <octomap_msgs/conversions.h>
#include <pcl/conversions.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <iostream>
#include <string>
#include <pcl/filters/filter.h>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

const char *LOGNAME = "env_tracker_node";
bool next_iteration = false;

void print4x4Matrix(const Eigen::Matrix4d &matrix) {
  printf("Rotation matrix :\n");
  printf("    | %6.3f %6.3f %6.3f | \n", matrix(0, 0), matrix(0, 1), matrix(0, 2));
  printf("R = | %6.3f %6.3f %6.3f | \n", matrix(1, 0), matrix(1, 1), matrix(1, 2));
  printf("    | %6.3f %6.3f %6.3f | \n", matrix(2, 0), matrix(2, 1), matrix(2, 2));
  printf("Translation vector :\n");
  printf("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix(0, 3), matrix(1, 3), matrix(2, 3));
}

template <typename T>
class Listener {
 public:
  explicit Listener(ros::NodeHandle &nh, std::string topic, int queue_size) {
    sub = nh.subscribe(topic, queue_size, &Listener::callback, this);
  }

  void callback(T const &msg) {
    std::lock_guard<std::mutex> lock(mutex);
    latest_msg = msg;
  }

  T get() {
    while (true) {
      {
        std::lock_guard<std::mutex> lock(mutex);
        if (latest_msg) {
          return latest_msg;
        }
      }
      ros::Duration(0.1).sleep();
    }
  }

  T latest_msg;
  ros::Subscriber sub;
  std::mutex mutex;
};

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "env_tracker_node");
  ros::NodeHandle nh;
  // subscribe to point cloud, at some frequency, get the latest one?
  Listener<sensor_msgs::PointCloud2ConstPtr> pc_listener(nh, "/kinect2_tripodA/qhd/points", 10);
  auto ps_pub = nh.advertise<moveit_msgs::PlanningScene>("/hdt_michigan/planning_scene", 10);

  std::string const robot_root_frame = "base_link";
  double const res = 0.01;

  if (argc != 2) {
    ROS_WARN_STREAM_NAMED(LOGNAME, "Usage: env_tracker_node path/to/obj.py");
    return EXIT_SUCCESS;
  }

  PointCloudT::Ptr cloud_in(new PointCloudT);
  const auto ply_filename = argv[1];
  auto const status = pcl::io::loadPLYFile(ply_filename, *cloud_in);
  if (status < 0) {
    ROS_FATAL_STREAM_NAMED(LOGNAME, "Failed to open file " << ply_filename);
  }

  ROS_INFO_STREAM_NAMED(LOGNAME, "Loaded file " << ply_filename << " (" << cloud_in->size() << " points)");

  ros::AsyncSpinner spinner(2);
  spinner.start();

  ros::Rate r(1);
  while (ros::ok()) {
    auto const pc_msg = pc_listener.get();
    pcl::PCLPointCloud2 pc_v2;
    pcl_conversions::toPCL(*pc_msg, pc_v2);
    PointCloudT pc_v1;
    pcl::fromPCLPointCloud2(pc_v2, pc_v1);

    pcl::Indices indices;
    PointCloudT pc_v1_nonan;
    pcl::removeNaNFromPointCloud(pc_v1, pc_v1_nonan, indices);

    ROS_WARN_STREAM_NAMED(LOGNAME, "pc_v1 size: " << pc_v1_nonan.size());

    octomap::OcTree tree(res);
    for (auto const &point : pc_v1_nonan.points) {
      octomap::point3d octo_point(point.x, point.y, point.z);
      ROS_DEBUG_STREAM_NAMED(LOGNAME, "point: " << point);
      tree.updateNode(octo_point, true);
    }

    //    auto points= boost::make_shared<PointCloudT>();
    //    pcl::transformPointCloud(points2_filtered, *points2_in_points1_frame, points2_to_points1.matrix());
    //
    //    Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();
    //    pcl::IterativeClosestPoint<PointT, PointT> icp;
    //    icp.setMaximumIterations(10);
    //    icp.setInputSource(points2_in_points1_frame);
    //    icp.setInputTarget(points1_filtered);
    //    icp.align(points2_icp);
    //
    //    if (icp.hasConverged()) {
    //        ROS_INFO_STREAM_NAMED(LOGNAME + ".icp", "ICP has converged, score is " << icp.getFitnessScore());
    //        const auto matrix = icp.getFinalTransformation().cast<double>();
    //        ROS_INFO_STREAM_NAMED(LOGNAME + ".icp", "Transformation:\n" << matrix);
    //    } else {
    //        ROS_ERROR_NAMED(LOGNAME, "ICP did not converge");
    //    }
    moveit_msgs::PlanningScene ps_update;
    auto const now = ros::Time::now();
    ps_update.is_diff = true;
    ps_update.robot_model_name = "husky";
    ps_update.world.octomap.header.stamp = now;
    ps_update.world.octomap.header.frame_id = robot_root_frame;

    octomap_msgs::binaryMapToMsg(tree, ps_update.world.octomap.octomap);
    ps_pub.publish(ps_update);

    r.sleep();
  }

  PointCloudT::Ptr cloud_tr(new PointCloudT);   // Transformed point cloud
  PointCloudT::Ptr cloud_icp(new PointCloudT);  // ICP output point cloud

  // Defining a rotation matrix and translation vector
  Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();

  //  // A rotation matrix (see https://en.wikipedia.org/wiki/Rotation_matrix)
  //  double theta = M_PI / 8;  // The angle of rotation in radians
  //  transformation_matrix(0, 0) = std::cos(theta);
  //  transformation_matrix(0, 1) = -sin(theta);
  //  transformation_matrix(1, 0) = sin(theta);
  //  transformation_matrix(1, 1) = std::cos(theta);
  //
  //  // A translation on Z axis (0.4 meters)
  //  transformation_matrix(2, 3) = 0.4;

  // Display in terminal the transformation matrix
  std::cout << "Applying this rigid transformation to: cloud_in -> cloud_icp" << std::endl;
  print4x4Matrix(transformation_matrix);

  // Executing the transformation
  pcl::transformPointCloud(*cloud_in, *cloud_icp, transformation_matrix);
  *cloud_tr = *cloud_icp;  // We backup cloud_icp into cloud_tr for later use

  auto const iterations = 10;
  // The Iterative Closest Point algorithm
  pcl::IterativeClosestPoint<PointT, PointT> icp;
  icp.setMaximumIterations(iterations);
  icp.setInputSource(cloud_icp);
  icp.setInputTarget(cloud_in);
  icp.align(*cloud_icp);

  // if (icp.hasConverged()) {
  std::cout << "\nICP has converged, score is " << icp.getFitnessScore() << std::endl;
  std::cout << "\nICP transformation " << iterations << " : cloud_icp -> cloud_in" << std::endl;
  transformation_matrix = icp.getFinalTransformation().cast<double>();
  print4x4Matrix(transformation_matrix);
}
