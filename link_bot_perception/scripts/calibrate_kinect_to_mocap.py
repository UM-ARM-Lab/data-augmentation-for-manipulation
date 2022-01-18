import argparse
import copy
import logging

import numpy as np
import open3d as o3d
import transformations

import rospy
from arc_utilities import ros_init
from arc_utilities.tf2wrapper import TF2Wrapper
from arm_robots.basic_3d_pose_interactive_marker import Basic3DPoseInteractiveMarker
from geometry_msgs.msg import Point, Pose, Quaternion, PointStamped
from lightweight_vicon_bridge.msg import MocapMarkerArray, MocapMarker
from ros_numpy import numpify, msgify
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker

logger = logging.getLogger(__file__)


def draw_registration_result(mocap_cloud, kinect_cloud, transformation):
    """ mocap will be shown in orange, kinect in blue """
    mocap_cloud_tmp = copy.deepcopy(mocap_cloud)
    kinect_cloud_tmp = copy.deepcopy(kinect_cloud)
    kinect_cloud_tmp_orig = copy.deepcopy(kinect_cloud)
    mocap_cloud_tmp.paint_uniform_color([1, 0.706, 0])
    kinect_cloud_tmp.paint_uniform_color([0, 0.651, 0.929])
    kinect_cloud_tmp_orig.paint_uniform_color([1, 0, 1])  # magenta for original
    kinect_cloud_tmp.transform(transformation)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([mocap_cloud_tmp, kinect_cloud_tmp_orig, kinect_cloud_tmp, frame])


@ros_init.with_ros("calibrate_kinect_to_mocap")
def main():
    np.set_printoptions(suppress=True, precision=4, linewidth=200)

    parser = argparse.ArgumentParser()
    parser.add_argument('kinect_tf_name', help='name of the kinect in mocap according to TF')
    parser.add_argument('m', type=int, help='number of camera positions to register from')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--load', type=str, help='load from this npz')

    args = parser.parse_args()

    pc_point_viz_pub = rospy.Publisher('calib_pc_points', Marker, queue_size=10)

    if args.load:
        data = np.load(args.load, allow_pickle=True)
        mocap_points_in_mocap_world = data['mocap_points']
        pc_points = data['pc_points']
        kinect2mocap_original = data['mocap_to_kinect_original']
    else:
        mocap_world_frame = 'mocap_world'

        radius = 0.006

        def make_marker(_):
            marker = Marker(type=Marker.SPHERE)
            marker.scale = Point(2 * radius, 2 * radius, 2 * radius)
            marker.color = ColorRGBA(0.5, 1.0, 0.5, 0.7)
            return marker

        im = Basic3DPoseInteractiveMarker(make_marker=make_marker, frame_id=args.kinect_tf_name)
        tf = TF2Wrapper()

        pc_points = []
        mocap_points_in_mocap_world = []
        for m_i in range(args.m):
            unlabeled_markers: MocapMarkerArray = rospy.wait_for_message("mocap_marker_tracking", MocapMarkerArray)
            marker: MocapMarker
            for marker in unlabeled_markers.markers:
                point_mocap_frame = numpify(marker.position)
                mocap_point_stamped = PointStamped(header=Header(frame_id=mocap_world_frame), point=marker.position)
                mocap_point_in_kinect_frame_msg = tf.transform_to_frame(mocap_point_stamped, args.kinect_tf_name)
                im.set_pose(Pose(position=mocap_point_in_kinect_frame_msg.point, orientation=Quaternion(w=1)))

                k = input("Save? [Y/n]")
                if k in ['n', 'N', 'q']:
                    print("Skipping...")
                    continue

                im_pose_kinect_frame = im.get_pose()
                im_point_kinect_frame = numpify(im_pose_kinect_frame.position)

                pc_points.append(im_point_kinect_frame)
                mocap_points_in_mocap_world.append(point_mocap_frame)

        kinect2mocap_original = tf.get_transform(mocap_world_frame, args.kinect_tf_name, time=rospy.Time.now())
        with open("calib.npz", 'wb') as file:
            np.savez(file, mocap_points=mocap_points_in_mocap_world, pc_points=pc_points,
                     mocap_to_kinect_original=kinect2mocap_original)

    if args.visualize:
        msg = Marker()
        msg.action = Marker.ADD
        msg.header.frame_id = 'kinect2_tripodA_rgb_optical_frame'
        msg.color.a = 1
        msg.color.r = 1
        msg.scale.x = 0.02
        msg.scale.y = 0.02
        msg.scale.z = 0.02
        msg.type = Marker.SPHERE_LIST
        msg.pose.orientation.w = 1
        for pc_point in pc_points:
            msg.points.append(msgify(Point, pc_point))
        pc_point_viz_pub.publish(msg)

    # transform the mocap points to be in the frame of the camera mocap
    def _tf(p):
        return (transformations.inverse_matrix(kinect2mocap_original) @ np.concatenate([p, [1]]))[:-1]

    mocap_points_in_mocap_kinect = [_tf(p) for p in mocap_points_in_mocap_world]

    pc_points_pcd = o3d.geometry.PointCloud()
    pc_points_pcd.points = o3d.utility.Vector3dVector(pc_points)

    mocap_points_pcd = o3d.geometry.PointCloud()
    mocap_points_pcd.points = o3d.utility.Vector3dVector(mocap_points_in_mocap_kinect)

    threshold = 0.35
    init = transformations.compose_matrix(angles=np.deg2rad([0, 0, 0]), translate=[0, 0, 0])
    mocap2pc = o3d.pipelines.registration.registration_icp(pc_points_pcd, mocap_points_pcd, threshold, init,
                                                           o3d.pipelines.registration.TransformationEstimationPointToPoint())

    print("transform from mocap to point cloud in kinect frame")
    trans = transformations.translation_from_matrix(mocap2pc.transformation)
    rot = transformations.euler_from_matrix(mocap2pc.transformation)
    roll, pitch, yaw = rot
    print('Copy This into the static_transform_publisher')
    print(f'{trans[0]:.5f} {trans[1]:.5f} {trans[2]:.5f} {yaw:.5f} {pitch:.5f} {roll:.5f}')
    print("NOTE: tf2_ros static_transform_publisher uses Yaw, Pitch, Roll so that's what is printed above")

    if args.visualize:
        draw_registration_result(mocap_points_pcd, pc_points_pcd, mocap2pc.transformation)


if __name__ == '__main__':
    main()
