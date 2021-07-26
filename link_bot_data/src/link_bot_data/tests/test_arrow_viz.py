from time import sleep

import numpy as np
import tensorflow as tf

import rospy
from link_bot_data.rviz_arrow import rviz_arrow
from link_bot_pycommon.pycommon import directions_3d
from visualization_msgs.msg import MarkerArray

if __name__ == '__main__':
    rospy.init_node('test_arrow_viz')
    action_viz_pub = rospy.Publisher("action_viz", MarkerArray, queue_size=10)
    r = 1
    g = 0
    b = 0
    a = 1

    N = 200
    pitch = tf.random.uniform([N], -np.pi, np.pi)
    yaw = tf.random.uniform([N], -np.pi, np.pi)
    directions = directions_3d(pitch, yaw)

    s1 = np.array([0.0, 0.0, 0.0])
    msg = MarkerArray()
    for idx, direction in enumerate(directions):
        s2 = s1 + direction * 0.1
        arrow = rviz_arrow(s1, s2, r, g, b, 1, idx=idx)
        msg.markers.append(arrow)

    for i in range(5):
        action_viz_pub.publish(msg)
        sleep(0.01)
