#include <iostream>
#include <tf2_eigen/tf2_eigen.h>
#include <string>

#include <ros/ros.h>
#include <pcl/conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <message_filters/subscriber.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf/transform_listener.h>

#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <pcl/point_types.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

const std::string LOGNAME = "merge_pointclouds";


auto filter(const PointCloudT::ConstPtr &points) {
    auto points_filtered = boost::make_shared<PointCloudT>();

    pcl::RadiusOutlierRemoval<PointT> outrem;
    outrem.setInputCloud(points);
    outrem.setRadiusSearch(0.8);
    outrem.setMinNeighborsInRadius(2);
    outrem.setKeepOrganized(true);
    outrem.filter(*points_filtered);

    return points_filtered;
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "merge_pointclouds");

    ros::NodeHandle nh;
    tf::TransformListener listener;

    auto const pub = nh.advertise<sensor_msgs::PointCloud2>("merged_points", 1);

    auto callback = [&](const sensor_msgs::PointCloud2ConstPtr &points1_msg,
                        const sensor_msgs::PointCloud2ConstPtr &points2_msg) -> void {

        const auto start = ros::WallTime::now();

        pcl::PCLPointCloud2 points1_v2;
        pcl::PCLPointCloud2 points2_v2;
        PointCloudT points1;
        PointCloudT points2;
        auto points1_nonan = boost::make_shared<PointCloudT>();
        auto points2_nonan = boost::make_shared<PointCloudT>();
        PointCloudT points2_icp;

        pcl_conversions::toPCL(*points1_msg, points1_v2);
        pcl_conversions::toPCL(*points2_msg, points2_v2);

        pcl::fromPCLPointCloud2(points1_v2, points1);
        pcl::fromPCLPointCloud2(points2_v2, points2);

        pcl::Indices indices;
        pcl::removeNaNFromPointCloud(points1, *points1_nonan, indices);
        pcl::removeNaNFromPointCloud(points2, *points2_nonan, indices);

        if (points1_nonan->empty()) {
            ROS_ERROR_STREAM("points1 is empty");
            return;
        }

        if (points2_nonan->empty()) {
            ROS_ERROR_STREAM("points2 is empty");
            return;
        }

//        const auto points1_filtered = filter(points1_nonan);
//        const auto points2_filtered = filter(points2_nonan);
        const auto points1_filtered = points1_nonan;
        const auto points2_filtered = points2_nonan;

        ROS_DEBUG_STREAM("filtered points " << points1_filtered->size() << ", " << points2_filtered->size());

        tf::StampedTransform points2_to_points1_stamped;
        try {
            listener.lookupTransform(points1_msg->header.frame_id, points2_msg->header.frame_id, ros::Time(0),
                                     points2_to_points1_stamped);
        }
        catch (tf::TransformException const &ex) {
            ROS_ERROR_STREAM_NAMED(LOGNAME, "Failed to lookup TF:" << ex.what());
        }

        geometry_msgs::TransformStamped points2_to_points1_stamped_msg;
        tf::transformStampedTFToMsg(points2_to_points1_stamped, points2_to_points1_stamped_msg);
        const auto points2_to_points1 = tf2::transformToEigen(points2_to_points1_stamped_msg);
        ROS_DEBUG_STREAM_NAMED(LOGNAME, "2to1 transform: \n" << points2_to_points1.matrix());

        auto points2_in_points1_frame = boost::make_shared<PointCloudT>();
        pcl::transformPointCloud(*points2_filtered, *points2_in_points1_frame, points2_to_points1.matrix());

        // The Iterative Closest Point algorithm
//        Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();
//        pcl::IterativeClosestPoint<PointT, PointT> icp;
//        icp.setMaximumIterations(10);
//        icp.setInputSource(points2_in_points1_frame);
//        icp.setInputTarget(points1_filtered);
//        icp.align(points2_icp);
//
//        if (icp.hasConverged()) {
//            ROS_INFO_STREAM_NAMED(LOGNAME + ".icp", "ICP has converged, score is " << icp.getFitnessScore());
//            const auto matrix = icp.getFinalTransformation().cast<double>();
//            ROS_INFO_STREAM_NAMED(LOGNAME + ".icp", "Transformation:\n" << matrix);
//        } else {
//            ROS_ERROR_NAMED(LOGNAME, "ICP did not converge");
//        }
        points2_icp = *points2_in_points1_frame;

        const auto merged_points = points2_icp + *points1_filtered;
        pcl::PCLPointCloud2 merged_points_v2;
        pcl::toPCLPointCloud2(merged_points, merged_points_v2);

        sensor_msgs::PointCloud2 output;
        pcl_conversions::fromPCL(merged_points_v2, output);
        output.header.frame_id = points1.header.frame_id;

        pub.publish(output);

        const auto end = ros::WallTime::now();
        const auto execution_time = (end - start).toNSec() * 1e-6;
        ROS_INFO_STREAM_NAMED(LOGNAME + ".dt", "dt (ms): " << execution_time);
    };

    message_filters::Subscriber<sensor_msgs::PointCloud2> sub1(nh, "points1", 10);
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub2(nh, "points2", 10);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::PointCloud2> ApproxSyncPolicy;
    message_filters::Synchronizer<ApproxSyncPolicy> sync(ApproxSyncPolicy(10), sub1, sub2);

    sync.registerCallback(boost::bind<void>(callback, _1, _2));


    ros::spin();

    return EXIT_SUCCESS;
}

