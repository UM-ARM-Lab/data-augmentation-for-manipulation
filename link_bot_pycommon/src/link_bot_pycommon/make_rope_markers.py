import rospy
from geometry_msgs.msg import Point
from link_bot_pycommon.marker_index_generator import marker_index_generator
from link_bot_pycommon.matplotlib_utils import adjust_lightness_msg
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker


def make_gripper_marker(position, id, color: ColorRGBA, label, type, s=0.02):
    gripper_marker = Marker()
    gripper_marker.action = Marker.ADD  # create or modify
    gripper_marker.type = type
    gripper_marker.header.frame_id = "world"
    gripper_marker.header.stamp = rospy.Time.now()
    gripper_marker.ns = label
    gripper_marker.id = id
    gripper_marker.scale.x = s
    gripper_marker.scale.y = s
    gripper_marker.scale.z = s
    gripper_marker.pose.position.x = position[0]
    gripper_marker.pose.position.y = position[1]
    gripper_marker.pose.position.z = position[2]
    gripper_marker.pose.orientation.w = 1
    gripper_marker.color = color
    return gripper_marker


def make_rope_marker(rope_points, frame_id, label, idx, color: ColorRGBA, s=0.02,
                     points_marker_type=Marker.SPHERE_LIST):
    ig = marker_index_generator(idx)
    lines = Marker()
    lines.action = Marker.ADD  # create or modify
    lines.type = Marker.LINE_STRIP
    lines.header.frame_id = frame_id
    lines.header.stamp = rospy.Time.now()
    lines.ns = label
    lines.id = next(ig)
    lines.pose.position.x = 0
    lines.pose.position.y = 0
    lines.pose.position.z = 0
    lines.pose.orientation.x = 0
    lines.pose.orientation.y = 0
    lines.pose.orientation.z = 0
    lines.pose.orientation.w = 1
    lines.scale.x = s
    lines.color = color

    points_marker = Marker()
    points_marker.action = Marker.ADD  # create or modify
    points_marker.type = points_marker_type
    points_marker.header.frame_id = frame_id
    points_marker.header.stamp = rospy.Time.now()
    points_marker.ns = label
    points_marker.id = next(ig)
    points_marker.scale.x = s
    points_marker.scale.y = s
    points_marker.scale.z = s
    points_marker.pose.position.x = 0
    points_marker.pose.position.y = 0
    points_marker.pose.position.z = 0
    points_marker.pose.orientation.x = 0
    points_marker.pose.orientation.y = 0
    points_marker.pose.orientation.z = 0
    points_marker.pose.orientation.w = 1
    points_marker.color = color
    for i, (x, y, z) in enumerate(rope_points):
        point = Point()
        point.x = x
        point.y = y
        point.z = z

        points_marker.points.append(point)
        lines.points.append(point)

    midpoint_sphere = Marker()
    midpoint_sphere.action = Marker.ADD  # create or modify
    midpoint_sphere.type = Marker.SPHERE
    midpoint_sphere.header.frame_id = frame_id
    midpoint_sphere.header.stamp = rospy.Time.now()
    midpoint_sphere.ns = label
    midpoint_sphere.id = next(ig)
    midpoint_sphere.scale.x = s * 1.2
    midpoint_sphere.scale.y = s * 1.2
    midpoint_sphere.scale.z = s * 1.2
    rope_midpoint = rope_points[int(rope_points.shape[0] / 2)]
    midpoint_sphere.pose.position.x = rope_midpoint[0]
    midpoint_sphere.pose.position.y = rope_midpoint[1]
    midpoint_sphere.pose.position.z = rope_midpoint[2]
    midpoint_sphere.pose.orientation.x = 0
    midpoint_sphere.pose.orientation.y = 0
    midpoint_sphere.pose.orientation.z = 0
    midpoint_sphere.pose.orientation.w = 1
    midpoint_sphere.color = adjust_lightness_msg(color, 0.8)

    first_point_text = Marker()
    first_point_text.action = Marker.ADD  # create or modify
    first_point_text.type = Marker.TEXT_VIEW_FACING
    first_point_text.header.frame_id = frame_id
    first_point_text.header.stamp = rospy.Time.now()
    first_point_text.ns = label
    first_point_text.id = next(ig)
    first_point_text.text = "0"
    first_point_text.scale.z = 0.015
    first_point_text.pose.position.x = rope_points[0, 0]
    first_point_text.pose.position.y = rope_points[0, 1]
    first_point_text.pose.position.z = rope_points[0, 2] + 0.015
    first_point_text.pose.orientation.x = 0
    first_point_text.pose.orientation.y = 0
    first_point_text.pose.orientation.z = 0
    first_point_text.pose.orientation.w = 1
    first_point_text.color.r = 1.0
    first_point_text.color.g = 1.0
    first_point_text.color.b = 1.0
    first_point_text.color.a = color.a
    return [points_marker, lines, midpoint_sphere, first_point_text]
