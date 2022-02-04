import argparse
import copy
import logging

import numpy as np
import open3d as o3d
import transformations

from arc_utilities import ros_init
from arc_utilities.tf2wrapper import TF2Wrapper

logger = logging.getLogger(__file__)


@ros_init.with_ros("calibrate_camera_to_mocap")
def main():
    np.set_printoptions(suppress=True, precision=4, linewidth=200)

    parser = argparse.ArgumentParser()
    parser.add_argument('camera_tf_name', help='name of the camera in mocap according to TF')
    parser.add_argument('m', help='number of fiducials to use for calibration')

    args = parser.parse_args()

    tf = TF2Wrapper()

    mocap_world_frame = 'mocap_world'
    world2camera_markers = tf.get_transform(mocap_world_frame, args.camera_tf_name)

    # transform the mocap points to be in the frame of the camera mocap
    def _tf(p):
        return (transformations.inverse_matrix(world2camera_markers) @ np.concatenate([p, [1]]))[:-1]

    world2tags = []
    for i in range(args.m):
        mocap2fiducial = tf.get_transform(mocap_world_frame, f"mocap_calib{i}_calib{i}")
        camera2fiducial = tf.get_transform(args.camera_tf_name, f"fiducial_{i}")
        fiducial2camera = transformations.inverse_matrix(camera2fiducial)
        mocap2camera_sensor_detected = mocap2fiducial @ fiducial2camera
        mocap2camera_markers = tf.get_transform(mocap_world_frame, args.camera_tf_name)
        mocap2camera_sensor_offset = np.linalg.solve(mocap2camera_markers, mocap2camera_sensor_detected)

        trans = transformations.translation_from_matrix(mocap2camera_sensor_offset)
        rot = transformations.euler_from_matrix(mocap2camera_sensor_offset)
        roll, pitch, yaw = rot
        print('Copy This into the static_transform_publisher')
        print(f'{trans[0]:.5f} {trans[1]:.5f} {trans[2]:.5f} {yaw:.5f} {pitch:.5f} {roll:.5f}')
        print("NOTE: tf2_ros static_transform_publisher uses Yaw, Pitch, Roll so that's what is printed above")


if __name__ == '__main__':
    main()
