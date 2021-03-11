#!/usr/bin/env python

import argparse
import sys

import rospy
from link_bot_pycommon.basic_3d_pose_marker import Basic3DPoseInteractiveMarker


def main():
    rospy.init_node("basic_controls")

    parser = argparse.ArgumentParser()
    parser.add_argument('--x', '-x', type=float, default=0)
    parser.add_argument('--y', '-y', type=float, default=0)
    parser.add_argument('--z', '-z', type=float, default=0)
    parser.add_argument('--shape', choices=['sphere', 'box'], default='box')

    args = parser.parse_args(rospy.myargv(sys.argv[1:]))

    i = Basic3DPoseInteractiveMarker(args.x, args.y, args.z, args.shape)

    rospy.spin()


if __name__ == "__main__":
    main()
