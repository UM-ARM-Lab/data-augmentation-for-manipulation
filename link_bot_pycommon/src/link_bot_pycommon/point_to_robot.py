import ros_numpy
from geometry_msgs.msg import PointStamped, Point


def point_to_root(robot, tf_wrapper, point, src: str):
    """

    Args:
        point: a np array of shape [3] representing an xyz point in src frame
        src:

    Returns: the point in the frame of the root of the robot

    """
    root_link = robot.robot_commander.get_root_link()
    msg = PointStamped()
    msg.header.frame_id = src
    msg.point = ros_numpy.msgify(Point, point)
    msg_transformed = tf_wrapper.transform_to_frame(msg, root_link)
    return ros_numpy.numpify(msg_transformed.point)
