// based on https://github.com/aosmundson/pcl-registration/blob/master/src/pipeline.cpp
#include <moveit_msgs/PlanningScene.h>
#include <octomap_msgs/conversions.h>
#include <pcl/conversions.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/point_types.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>

#include <iostream>
#include <string>

double const res = 0.01;

// --------------------
// -----Parameters-----
// SIFT Keypoint parameters
const float min_scale = 0.01f;      // the standard deviation of the smallest scale in the scale space
const int n_octaves = 3;            // the number of octaves (i.e. doublings of scale) to compute
const int n_scales_per_octave = 4;  // the number of scales to compute within each octave
const float min_contrast = 0.001f;  // the minimum contrast required for detection

// Sample Consensus Initial Alignment parameters (explanation below)
const float min_sample_dist = 0.625f;
const float max_correspondence_dist = 0.05f;
const int nr_iters = 500;

// ICP parameters (explanation below)
const float max_correspondence_distance = 0.1f;
const float outlier_rejection_threshold = 0.1f;
const float transformation_epsilon = 0;
const int max_iterations = 100;
// --------------------
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef PointCloud::Ptr PointCloudPtr;
typedef pcl::PointCloud<pcl::PointWithScale>::Ptr PointCloudScalePtr;
typedef pcl::FPFHSignature33 LocalDescriptorT;
typedef pcl::PointCloud<LocalDescriptorT>::Ptr LocalDescriptorsScalePtr;

const char *LOGNAME = "env_tracker_node";
bool next_iteration = false;

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

void debug_pub(ros::Publisher const &pub, PointCloud const &pc, std::string const &frame_id) {
  sensor_msgs::PointCloud2 msg;
  pcl::PCLPointCloud2 pc2_tmp;
  pcl::toPCLPointCloud2(pc, pc2_tmp);
  pcl_conversions::fromPCL(pc2_tmp, msg);
  msg.header.frame_id = frame_id;
  pub.publish(msg);
}

Eigen::Matrix4f computeInitialAlignment(const PointCloudScalePtr &source_points,
                                        const LocalDescriptorsScalePtr &source_descriptors,
                                        const PointCloudScalePtr &target_points,
                                        const LocalDescriptorsScalePtr &target_descriptors, float min_sample_distance,
                                        float max_correspondence_distance, int nr_iterations) {
  pcl::SampleConsensusInitialAlignment<pcl::PointWithScale, pcl::PointWithScale, LocalDescriptorT> sac_ia;
  sac_ia.setMinSampleDistance(min_sample_distance);
  sac_ia.setMaxCorrespondenceDistance(max_correspondence_distance);
  sac_ia.setMaximumIterations(nr_iterations);

  sac_ia.setInputCloud(source_points);
  sac_ia.setSourceFeatures(source_descriptors);

  sac_ia.setInputTarget(target_points);
  sac_ia.setTargetFeatures(target_descriptors);

  pcl::PointCloud<pcl::PointWithScale> registration_output;
  sac_ia.align(registration_output);

  return sac_ia.getFinalTransformation();
}

/* Use IterativeClosestPoint to find a precise alignment from the source cloud to the target cloud,
 * starting with an intial guess
 * Inputs:
 *   source_points
 *     The "source" points, i.e., the points that must be transformed to align with the target point cloud
 *   target_points
 *     The "target" points, i.e., the points to which the source point cloud will be aligned
 *   intial_alignment
 *     An initial estimate of the transformation matrix that aligns the source points to the target points
 *   max_correspondence_distance
 *     A threshold on the distance between any two corresponding points.  Any corresponding points that are further
 *     apart than this threshold will be ignored when computing the source-to-target transformation
 *   outlier_rejection_threshold
 *     A threshold used to define outliers during RANSAC outlier rejection
 *   transformation_epsilon
 *     The smallest iterative transformation allowed before the algorithm is considered to have converged
 *   max_iterations
 *     The maximum number of ICP iterations to perform
 * Return: A transformation matrix that will precisely align the points in source to the points in target
 */
Eigen::Matrix4f refineAlignment(const PointCloudPtr &source_points, const PointCloudPtr &target_points,
                                const Eigen::Matrix4f initial_alignment, float max_correspondence_distance,
                                float outlier_rejection_threshold, float transformation_epsilon, float max_iterations) {
  pcl::IterativeClosestPoint<PointT, PointT> icp;
  icp.setMaxCorrespondenceDistance(max_correspondence_distance);
  icp.setRANSACOutlierRejectionThreshold(outlier_rejection_threshold);
  icp.setTransformationEpsilon(transformation_epsilon);
  icp.setMaximumIterations(max_iterations);

  auto source_points_transformed = boost::make_shared<PointCloud>();
  pcl::transformPointCloud(*source_points, *source_points_transformed, initial_alignment);

  icp.setInputCloud(source_points_transformed);
  icp.setInputTarget(target_points);

  PointCloud registration_output;
  icp.align(registration_output);

  if (icp.hasConverged()) {
    ROS_INFO_STREAM_NAMED(LOGNAME + ".icp", "ICP has converged, score is " << icp.getFitnessScore());
  } else {
    ROS_ERROR_NAMED(LOGNAME, "ICP did not converge");
  }

  return icp.getFinalTransformation() * initial_alignment;
}

PointCloud icp2(PointCloud const &src_pc, PointCloud const &target_pc) {
  auto src_pc_ptr = boost::make_shared<PointCloud>(src_pc);
  auto target_pc_ptr = boost::make_shared<PointCloud>(target_pc);

  pcl::NormalEstimation<PointT, pcl::PointNormal> ne;
  auto src_normals_ptr = boost::make_shared<pcl::PointCloud<pcl::PointNormal>>();
  pcl::PointCloud<pcl::PointNormal> &src_normals = *src_normals_ptr;
  auto tree_xyz = boost::make_shared<pcl::search::KdTree<PointT>>();
  ne.setInputCloud(src_pc_ptr);
  ne.setSearchMethod(tree_xyz);
  ne.setRadiusSearch(0.05);
  ne.compute(*src_normals_ptr);
  for (size_t i = 0; i < src_normals.points.size(); ++i) {
    src_normals.points[i].x = src_pc.points[i].x;
    src_normals.points[i].y = src_pc.points[i].y;
    src_normals.points[i].z = src_pc.points[i].z;
  }

  auto tar_normals_ptr = boost::make_shared<pcl::PointCloud<pcl::PointNormal>>();
  pcl::PointCloud<pcl::PointNormal> &tar_normals = *tar_normals_ptr;
  ne.setInputCloud(target_pc_ptr);
  ne.compute(*tar_normals_ptr);
  for (size_t i = 0; i < tar_normals.points.size(); ++i) {
    tar_normals.points[i].x = target_pc.points[i].x;
    tar_normals.points[i].y = target_pc.points[i].y;
    tar_normals.points[i].z = target_pc.points[i].z;
  }

  // Estimate the SIFT keypoints
  pcl::SIFTKeypoint<pcl::PointNormal, pcl::PointWithScale> sift;
  auto src_keypoints_ptr = boost::make_shared<pcl::PointCloud<pcl::PointWithScale>>();
  pcl::PointCloud<pcl::PointWithScale> &src_keypoints = *src_keypoints_ptr;
  auto tree_normal = boost::make_shared<pcl::search::KdTree<pcl::PointNormal>>();
  sift.setSearchMethod(tree_normal);
  sift.setScales(min_scale, n_octaves, n_scales_per_octave);
  sift.setMinimumContrast(min_contrast);
  sift.setInputCloud(src_normals_ptr);
  sift.compute(src_keypoints);

  ROS_DEBUG_STREAM_NAMED(LOGNAME, "Found " << src_keypoints.points.size() << " SIFT keypoints in source cloud");

  auto tar_keypoints_ptr = boost::make_shared<pcl::PointCloud<pcl::PointWithScale>>();
  pcl::PointCloud<pcl::PointWithScale> &tar_keypoints = *tar_keypoints_ptr;
  sift.setInputCloud(tar_normals_ptr);
  sift.compute(tar_keypoints);

  ROS_DEBUG_STREAM_NAMED(LOGNAME, "Found " << tar_keypoints.points.size() << " SIFT keypoints in target cloud");

  // Extract FPFH features from SIFT keypoints
  auto src_keypoints_xyz = boost::make_shared<PointCloud>();
  pcl::copyPointCloud(src_keypoints, *src_keypoints_xyz);
  pcl::FPFHEstimation<PointT, pcl::PointNormal, pcl::FPFHSignature33> fpfh;
  fpfh.setSearchSurface(src_pc_ptr);
  fpfh.setInputCloud(src_keypoints_xyz);
  fpfh.setInputNormals(src_normals_ptr);
  fpfh.setSearchMethod(tree_xyz);
  auto src_features_ptr = boost::make_shared<pcl::PointCloud<pcl::FPFHSignature33>>();
  pcl::PointCloud<pcl::FPFHSignature33> &src_features = *src_features_ptr;
  fpfh.setRadiusSearch(0.05);
  fpfh.compute(src_features);
  ROS_DEBUG_STREAM_NAMED(LOGNAME, "Computed " << src_features.size() << " FPFH features for source cloud");

  auto tar_keypoints_xyz = boost::make_shared<PointCloud>();
  pcl::copyPointCloud(tar_keypoints, *tar_keypoints_xyz);
  fpfh.setSearchSurface(target_pc_ptr);
  fpfh.setInputCloud(tar_keypoints_xyz);
  fpfh.setInputNormals(tar_normals_ptr);
  auto tar_features_ptr = boost::make_shared<pcl::PointCloud<pcl::FPFHSignature33>>();
  pcl::PointCloud<pcl::FPFHSignature33> &tar_features = *tar_features_ptr;
  fpfh.compute(tar_features);
  ROS_DEBUG_STREAM_NAMED(LOGNAME, "Computed " << tar_features.size() << " FPFH features for target cloud");

  Eigen::Matrix4f tform = Eigen::Matrix4f::Identity();
  tform = computeInitialAlignment(src_keypoints_ptr, src_features_ptr, tar_keypoints_ptr, tar_features_ptr,
                                  min_sample_dist, max_correspondence_dist, nr_iters);

  tform = refineAlignment(src_pc_ptr, target_pc_ptr, tform, max_correspondence_distance, outlier_rejection_threshold,
                          transformation_epsilon, max_iterations);

  auto const transformed_pc_ptr = boost::make_shared<PointCloud>();
  PointCloud &transformed_pc = *transformed_pc_ptr;
  pcl::transformPointCloud(src_pc, transformed_pc, tform);

  return transformed_pc;
};

PointCloud icp(PointCloud const &src_pc, PointCloud const &target_pc) {
  PointCloud src_pc_aligned;
  auto src_pc_ptr = boost::make_shared<PointCloud>(src_pc);
  auto target_pc_ptr = boost::make_shared<PointCloud>(target_pc);
  pcl::IterativeClosestPoint<PointT, PointT> icp;
  icp.setMaximumIterations(max_iterations);
//  icp.setMaxCorrespondenceDistance(max_correspondence_distance);
//  icp.setRANSACOutlierRejectionThreshold(outlier_rejection_threshold);
//  icp.setTransformationEpsilon(transformation_epsilon);
  icp.setUseReciprocalCorrespondences(true);
  icp.setInputSource(src_pc_ptr);
  icp.setInputTarget(target_pc_ptr);
  Eigen::Isometry3f initial_guess{Eigen::Isometry3f::Identity()};
  initial_guess.translation().x() += 0.4;
  initial_guess.translation().y() += 0.6;
  initial_guess.translation().z() -= 0.3;
  icp.align(src_pc_aligned, initial_guess.matrix());

  if (icp.hasConverged()) {
    ROS_INFO_STREAM_NAMED(LOGNAME + ".icp", "ICP has converged, score is " << icp.getFitnessScore());
  } else {
    ROS_ERROR_NAMED(LOGNAME, "ICP did not converge");
  }

  return src_pc_aligned;
};

PointCloud remove_outliers(PointCloud const &pc) {
  PointCloud pc_out;
  pcl::StatisticalOutlierRemoval<PointT> sor;
  auto const pc_ptr = boost::make_shared<PointCloud>(pc);
  sor.setInputCloud(pc_ptr);
  sor.setMeanK(50);
  sor.setStddevMulThresh(0.1);
  sor.filter(pc_out);

  return pc_out;
}

auto downsample(PointCloud const &pc, double const res) {
  PointCloud pc_downsampled;
  pcl::VoxelGrid<PointT> filter;
  PointCloudPtr pc_ptr = boost::make_shared<PointCloud>(pc);
  filter.setInputCloud(pc_ptr);
  filter.setLeafSize(res, res, res);
  filter.filter(pc_downsampled);
  return pc_downsampled;
};

auto simplify(PointCloud const &pc, double const res) {
  return remove_outliers(downsample(remove_outliers(pc), 2 * res));
}

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "env_tracker_node");
  ros::NodeHandle nh;
  // subscribe to point cloud, at some frequency, get the latest one?
  Listener<sensor_msgs::PointCloud2ConstPtr> pc_listener(nh, "/kinect2_tripodA/qhd/points", 10);
  auto ps_pub = nh.advertise<moveit_msgs::PlanningScene>("/hdt_michigan/planning_scene", 10);
  auto icp_src_pub = nh.advertise<sensor_msgs::PointCloud2>("icp_src", 10);
  auto icp_target_pub = nh.advertise<sensor_msgs::PointCloud2>("icp_target", 10);
  auto out_pub = nh.advertise<sensor_msgs::PointCloud2>("icp_out", 10);

  std::string const robot_root_frame = "base_link";
  std::string const camera_frame = "kinect2_tripodA_rgb_optical_frame";

  if (argc != 2) {
    ROS_WARN_STREAM_NAMED(LOGNAME, "Usage: env_tracker_node path/to/obj.py");
    return EXIT_SUCCESS;
  }

  PointCloud env_pc;
  const auto ply_filename = argv[1];
  auto const status = pcl::io::loadPLYFile(ply_filename, env_pc);
  if (status < 0) {
    ROS_FATAL_STREAM_NAMED(LOGNAME, "Failed to open file " << ply_filename);
  }

  auto const env_pc_downsampled = simplify(env_pc, res);
  ROS_INFO_STREAM_NAMED(LOGNAME, "Loaded file " << ply_filename << " (" << env_pc_downsampled.size() << " points)");

  ros::AsyncSpinner spinner(1);
  spinner.start();

  tf2_ros::Buffer tf_buffer;
  tf2_ros::TransformListener tfListener(tf_buffer);

  auto get_observed_pc = [&](Eigen::Isometry3d const &camera2robot_root) {
    auto const pc_msg = pc_listener.get();
    pcl::PCLPointCloud2 pc2_tmp;
    pcl_conversions::toPCL(*pc_msg, pc2_tmp);
    PointCloud pc;
    pcl::fromPCLPointCloud2(pc2_tmp, pc);

    pcl::Indices indices;
    PointCloud pc_nonan;
    pcl::removeNaNFromPointCloud(pc, pc_nonan, indices);

    PointCloud pc_robot_frame;
    pcl::transformPointCloud(pc_nonan, pc_robot_frame, camera2robot_root.cast<float>());

    PointCloud pc_robot_frame_cropped;
    pcl::CropBox<PointT> box_filter;
    box_filter.setMin(Eigen::Vector4f(-1, -1, -0.15, 1.0));
    box_filter.setMax(Eigen::Vector4f(2, 1, 2, 1.0));
    PointCloudPtr pc_robot_frame_ptr = boost::make_shared<PointCloud>(pc_robot_frame);
    box_filter.setInputCloud(pc_robot_frame_ptr);
    box_filter.filter(pc_robot_frame_cropped);

    auto pc_downsampled = simplify(pc_robot_frame_cropped, res);

    return pc_downsampled;
  };

  auto publish = [&](PointCloud const &final_pc) {
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
    debug_pub(out_pub, final_pc, robot_root_frame);
  };

  ros::Rate r(1);
  while (ros::ok()) {
    auto const camera2robot_root_msg =
        tf_buffer.lookupTransform(robot_root_frame, camera_frame, ros::Time(0), ros::Duration(5));
    auto const camera2robot_root = tf2::transformToEigen(camera2robot_root_msg);

    auto const observed_pc = get_observed_pc(camera2robot_root);
    ROS_WARN_STREAM_NAMED(LOGNAME, "pc size: " << observed_pc.size());

    PointCloud env_pc_robot_frame;
    pcl::transformPointCloud(env_pc_downsampled, env_pc_robot_frame, camera2robot_root.cast<float>());

    debug_pub(icp_src_pub, env_pc_robot_frame, robot_root_frame);
    debug_pub(icp_target_pub, observed_pc, robot_root_frame);
    auto const env_pc_aligned = icp(env_pc_robot_frame, observed_pc);

    publish(env_pc_aligned);

    r.sleep();
  }
}
