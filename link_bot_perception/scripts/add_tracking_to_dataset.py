#!/usr/bin/env python
import argparse
import pathlib
from typing import Dict

import colorama
import numpy as np

import ros_numpy
import rospy
from arc_utilities.listener import Listener
from arc_utilities.tf2wrapper import TF2Wrapper
from link_bot_data.dynamics_dataset import DynamicsDatasetLoader
from link_bot_data.modify_dataset import modify_dataset
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.constants import KINECT_MAX_DEPTH
from link_bot_pycommon.ros_pycommon import transform_points_to_robot_frame
from rospy_message_converter import message_converter
from sensor_msgs.msg import PointCloud2, Image, CameraInfo, JointState
from tf import transformations


def main():
    colorama.init(autoreset=True)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')

    args = parser.parse_args()

    rospy.init_node("modify_dynamics_dataset")

    outdir = args.dataset_dir.parent / f"{args.dataset_dir.name}+cdcpd"

    cdcpd_listener = Listener("cdcpd/output", PointCloud2)

    color_pub = rospy.Publisher("kinect2/qhd/image_color_rect", Image, queue_size=10)
    depth_pub = rospy.Publisher("kinect2/qhd/image_depth_rect", Image, queue_size=10)
    info_pub = rospy.Publisher("kinect2/qhd/camera_info", CameraInfo, queue_size=10)
    gripper_configs_pub = rospy.Publisher("/victor/joint_states_viz", JointState, queue_size=10)

    tf = TF2Wrapper()

    def _process_example(dataset: DynamicsDatasetLoader, example: Dict):
        # publish the data CDCPD needs
        rgbd = example['rgbd']
        rgb = rgbd[:, :, :, :-1]
        depth = rgbd[:, :, :, -1]
        bgr = np.flip(rgb, axis=-1)

        tracked_rope = []
        for t in range(rgbd.shape[0]):
            bgr_t = bgr[t].numpy().astype(np.uint8)
            depth_t = depth[t].numpy()

            color_msg = ros_numpy.msgify(Image, bgr_t, encoding='bgr8')
            color_msg.header.frame_id = 'kinect2_rgb_optical_frame'

            depth_t[depth_t == KINECT_MAX_DEPTH] = np.nan
            depth_msg = ros_numpy.msgify(Image, depth_t, encoding='32FC1')
            depth_msg.header.frame_id = 'blehhhhh'

            kinect_params = dataset.scenario_metadata['kinect_params']
            world_to_rgb_optical_frame_mat = np.array(dataset.scenario_metadata['world_to_rgb_optical_frame'])
            world_to_rgb_optical_frame_q = transformations.quaternion_from_matrix(world_to_rgb_optical_frame_mat)
            world_to_rgb_optical_frame_xyz = transformations.translation_from_matrix(world_to_rgb_optical_frame_mat)
            camera_info_msg = message_converter.convert_dictionary_to_ros_message('sensor_msgs/CameraInfo',
                                                                                  kinect_params)

            # rotation of gripper not used
            left_gripper_translation_t = example['left_gripper'][t].numpy()
            right_gripper_translation_t = example['right_gripper'][t].numpy()

            def send():
                now = rospy.Time.now()

                tf.send_transform(translation=left_gripper_translation_t,
                                  quaternion=[0, 0, 0, 1.0],
                                  parent='world', child='left_tool',
                                  is_static=False, time=now)
                tf.send_transform(translation=right_gripper_translation_t,
                                  quaternion=[0, 0, 0, 1.0],
                                  parent='world', child='right_tool',
                                  is_static=False, time=now)
                tf.send_transform(translation=world_to_rgb_optical_frame_xyz,
                                  quaternion=world_to_rgb_optical_frame_q,
                                  parent='world', child='kinect2_rgb_optical_frame',
                                  is_static=False, time=now)

                color_msg.header.stamp = now
                depth_msg.header.stamp = now
                camera_info_msg.header.stamp = now

                color_pub.publish(color_msg)
                depth_pub.publish(depth_msg)
                info_pub.publish(camera_info_msg)

            send()
            # rospy.sleep(1.0)

            # get the response
            cdcpd_msg: PointCloud2 = cdcpd_listener.get()
            points = transform_points_to_robot_frame(tf, cdcpd_msg)
            cdcpd_vector = points.flatten()
            tracked_rope.append(cdcpd_vector)

        example['rope'] = tracked_rope
        yield example

    hparams_update = {}

    dataset = DynamicsDatasetLoader([args.dataset_dir])
    modify_dataset(dataset_dir=args.dataset_dir,
                   dataset=dataset,
                   outdir=outdir,
                   process_example=_process_example,
                   hparams_update=hparams_update)


if __name__ == '__main__':
    main()
