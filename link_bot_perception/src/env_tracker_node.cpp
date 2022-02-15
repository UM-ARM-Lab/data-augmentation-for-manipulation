#include <moveit_msgs/PlanningScene.h>
#include <octomap_msgs/conversions.h>
#include <pcl/conversions.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>

#include <iostream>
#include <string>

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

void debug_pub(ros::Publisher const &pub, PointCloudT const &pc, std::string const &frame_id) {
  sensor_msgs::PointCloud2 msg;
  pcl::PCLPointCloud2 pc2_tmp;
  pcl::toPCLPointCloud2(pc, pc2_tmp);
  pcl_conversions::fromPCL(pc2_tmp, msg);
  msg.header.frame_id = frame_id;
  pub.publish(msg);
}
int main(int argc, char *argv[]) {
  ros::init(argc, argv, "env_tracker_node");
  ros::NodeHandle nh;
  // subscribe to point cloud, at some frequency, get the latest one?
  Listener<sensor_msgs::PointCloud2ConstPtr> pc_listener(nh, "/kinect2_tripodA/qhd/points", 10);
  auto ps_pub = nh.advertise<moveit_msgs::PlanningScene>("/hdt_michigan/planning_scene", 10);
  auto icp_src_pub = nh.advertise<sensor_msgs::PointCloud2>("icp_src", 10);
  auto icp_target_pub = nh.advertise<sensor_msgs::PointCloud2>("icp_target", 10);

  std::string const robot_root_frame = "base_link";
  std::string const camera_frame = "kinect2_tripodA_rgb_optical_frame";
  double const res = 0.01;
  int nr_iterations = 50;

  if (argc != 2) {
    ROS_WARN_STREAM_NAMED(LOGNAME, "Usage: env_tracker_node path/to/obj.py");
    return EXIT_SUCCESS;
  }

  PointCloudT env_pc;
  const auto ply_filename = argv[1];
  auto const status = pcl::io::loadPLYFile(ply_filename, env_pc);
  if (status < 0) {
    ROS_FATAL_STREAM_NAMED(LOGNAME, "Failed to open file " << ply_filename);
  }

  auto downsample = [&](PointCloudT const &pc) {
    PointCloudT pc_downsampled;
    pcl::VoxelGrid<PointT> filter;
    PointCloudT::Ptr pc_ptr = boost::make_shared<PointCloudT>(pc);
    filter.setInputCloud(pc_ptr);
    filter.setLeafSize(res, res, res);
    filter.filter(pc_downsampled);
    return pc_downsampled;
  };

  auto const env_pc_downsampled = downsample(env_pc);
  ROS_INFO_STREAM_NAMED(LOGNAME, "Loaded file " << ply_filename << " (" << env_pc_downsampled.size() << " points)");

  ros::AsyncSpinner spinner(1);
  spinner.start();

  tf2_ros::Buffer tf_buffer;
  tf2_ros::TransformListener tfListener(tf_buffer);
  Eigen::Matrix4f last_guess{Eigen::Matrix4f::Identity()};

  auto get_observed_pc = [&](Eigen::Isometry3d const &camera2robot_root) {
    auto const pc_msg = pc_listener.get();
    pcl::PCLPointCloud2 pc2_tmp;
    pcl_conversions::toPCL(*pc_msg, pc2_tmp);
    PointCloudT pc;
    pcl::fromPCLPointCloud2(pc2_tmp, pc);

    pcl::Indices indices;
    PointCloudT pc_nonan;
    pcl::removeNaNFromPointCloud(pc, pc_nonan, indices);

    PointCloudT pc_robot_frame;
    pcl::transformPointCloud(pc_nonan, pc_robot_frame, camera2robot_root.cast<float>());

    PointCloudT pc_robot_frame_cropped;
    pcl::CropBox<PointT> box_filter;
    box_filter.setMin(Eigen::Vector4f(-1, -1, -0.15, 1.0));
    box_filter.setMax(Eigen::Vector4f(2, 1, 2, 1.0));
    PointCloudT::Ptr pc_robot_frame_ptr = boost::make_shared<PointCloudT>(pc_robot_frame);
    box_filter.setInputCloud(pc_robot_frame_ptr);
    box_filter.filter(pc_robot_frame_cropped);

    auto pc_downsampled = downsample(pc_robot_frame_cropped);

    return pc_downsampled;
  };

  auto icp = [&](PointCloudT const &src_pc, PointCloudT const &target_pc) {
    debug_pub(icp_src_pub, src_pc, robot_root_frame);
    debug_pub(icp_target_pub, target_pc, robot_root_frame);

    PointCloudT src_pc_aligned;
    auto src_pc_ptr = boost::make_shared<PointCloudT>(src_pc);
    auto target_pc_ptr = boost::make_shared<PointCloudT>(target_pc);
    pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setMaximumIterations(nr_iterations);
    icp.setUseReciprocalCorrespondences(true);
    icp.setInputSource(src_pc_ptr);
    icp.setInputTarget(target_pc_ptr);
    icp.align(src_pc_aligned, last_guess);
    last_guess = icp.getFinalTransformation();

    if (icp.hasConverged()) {
      ROS_INFO_STREAM_NAMED(LOGNAME + ".icp", "ICP has converged, score is " << icp.getFitnessScore());
      const auto matrix = icp.getFinalTransformation();
      ROS_INFO_STREAM_NAMED(LOGNAME + ".icp", "Transformation:\n" << matrix);
    } else {
      ROS_ERROR_NAMED(LOGNAME, "ICP did not converge");
    }

    return src_pc_aligned;
  };

  auto publish = [&](PointCloudT const &final_pc) {
    octomap::OcTree tree(res);
    for (auto const &point : final_pc) {
      octomap::point3d octo_point(point.x, point.y, point.z);
      tree.updateNode(octo_point, true);
    }

    moveit_msgs::PlanningScene ps_update;
    auto const now = ros::Time::now();
    ps_update.is_diff = true;
    ps_update.robot_model_name = "husky";
    ps_update.world.octomap.header.stamp = now;
    ps_update.world.octomap.header.frame_id = robot_root_frame;

    octomap_msgs::binaryMapToMsg(tree, ps_update.world.octomap.octomap);
    ps_pub.publish(ps_update);
  };

  ros::Rate r(1);
  while (ros::ok()) {
    auto const camera2robot_root_msg =
        tf_buffer.lookupTransform(robot_root_frame, camera_frame, ros::Time(0), ros::Duration(5));
    auto const camera2robot_root = tf2::transformToEigen(camera2robot_root_msg);

    auto const observed_pc = get_observed_pc(camera2robot_root);
    ROS_WARN_STREAM_NAMED(LOGNAME, "pc size: " << observed_pc.size());

    PointCloudT env_pc_robot_frame;
    pcl::transformPointCloud(env_pc_downsampled, env_pc_robot_frame, camera2robot_root.cast<float>());

    auto const env_pc_aligned = icp(env_pc_robot_frame, observed_pc);

    publish(env_pc_aligned);

    r.sleep();
  }
}
