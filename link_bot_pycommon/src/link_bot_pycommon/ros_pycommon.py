import numpy as np

import ros_numpy
import rospy
from arc_utilities.listener import Listener
from arc_utilities.tf2wrapper import TF2Wrapper
from peter_msgs.srv import Position3DEnable, GetPosition3D, Position3DAction
from rosgraph.names import ns_join
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_srvs.srv import Empty
from tf2_sensor_msgs import tf2_sensor_msgs


def make_movable_object_services(object_name):
    return {
        'enable':       rospy.ServiceProxy(f'{object_name}/enable', Position3DEnable),
        'get_position': rospy.ServiceProxy(f'{object_name}/get', GetPosition3D),
        'set':          rospy.ServiceProxy(f'{object_name}/set', Position3DAction),
        'move':         rospy.ServiceProxy(f'{object_name}/move', Position3DAction),
        'stop':         rospy.ServiceProxy(f'{object_name}/stop', Empty),
    }


def publish_color_image(pub: rospy.Publisher, x):
    color = x.astype(np.uint8)
    color_viz_msg = ros_numpy.msgify(Image, color, encoding="rgb8")
    pub.publish(color_viz_msg)


def publish_depth_image(pub: rospy.Publisher, x):
    depth_viz_msg = ros_numpy.msgify(Image, x, encoding="32FC1")
    pub.publish(depth_viz_msg)


def get_camera_params(camera_name: str):
    camera_params_topic_name = ns_join(ns_join(camera_name, 'qhd'), "camera_info")
    camera_params_listener = Listener(camera_params_topic_name, CameraInfo)
    camera_params: CameraInfo = camera_params_listener.get()
    return camera_params


def transform_points_to_robot_frame(tf: TF2Wrapper, cdcpd_msg: PointCloud2, robot_frame_id: str = 'robot_root'):
    """ transform into robot-frame """
    transform = tf.get_transform_msg(robot_frame_id, cdcpd_msg.header.frame_id)
    cdcpd_points_robot_frame = tf2_sensor_msgs.do_transform_cloud(cdcpd_msg, transform)
    cdcpd_points_array = ros_numpy.numpify(cdcpd_points_robot_frame)
    x = cdcpd_points_array['x']
    y = cdcpd_points_array['y']
    z = cdcpd_points_array['z']
    points = np.stack([x, y, z], axis=-1)
    return points


