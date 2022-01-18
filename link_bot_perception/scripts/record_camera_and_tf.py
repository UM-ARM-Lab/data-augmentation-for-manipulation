#!/usr/bin/env python
import pathlib

from PIL import Image
import numpy as np
import argparse
import sys

import ros_numpy
import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from geometry_msgs.msg import Transform
from sensor_msgs.msg import Image as ImageMsg


def main():
    rospy.init_node('record_camera_and_tf')

    parser = argparse.ArgumentParser()
    parser.add_argument('name', help='used to create a directory')
    parser.add_argument('parent')
    parser.add_argument('child')

    tf = TF2Wrapper()
    idx = 0
    args = parser.parse_args(rospy.myargv(sys.argv[1:]))
    root_dir = pathlib.Path('camera_and_tf') / args.name
    images_dir = root_dir / 'images'
    transforms_dir = root_dir / 'transforms'
    images_dir.mkdir(parents=True, exist_ok=True)
    transforms_dir.mkdir(parents=True, exist_ok=True)

    def callback(img_msg: ImageMsg):
        nonlocal idx
        idx += 1
        mocap_transform = tf.get_transform(parent=args.parent, child=args.child)

        img_filename = root_dir / 'images' / f'frame{idx}.png'
        transform_filename = root_dir / 'transforms' / f'transform{idx}.txt'

        np_img = ros_numpy.numpify(img_msg)
        r = np_img[:, :, 2]
        g = np_img[:, :, 1]
        b = np_img[:, :, 0]
        np_img = np.stack([r, g, b], axis=-1)
        pil_img = Image.fromarray(np_img)
        pil_img.save(img_filename)

        np.savetxt(transform_filename, mocap_transform)

        print(f"Saved {img_filename.as_posix()}, {transform_filename.as_posix()}")

    sub = rospy.Subscriber('image', ImageMsg, callback)

    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        r.sleep()


if __name__ == '__main__':
    main()
