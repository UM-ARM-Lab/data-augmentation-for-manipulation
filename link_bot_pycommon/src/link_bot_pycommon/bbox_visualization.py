import rospy

try:
    from jsk_recognition_msgs.msg import BoundingBox
except ImportError:
    rospy.logwarn("ignoring failed import of BBox message")


def grid_to_bbox(rows: int,
                 cols: int,
                 channels: int,
                 resolution: float):
    xsize = cols * resolution
    ysize = rows * resolution
    zsize = channels * resolution

    cx = xsize / 2
    cy = ysize / 2
    cz = zsize / 2

    bbox_msg = BoundingBox()
    bbox_msg.pose.position.x = cx
    bbox_msg.pose.position.y = cy
    bbox_msg.pose.position.z = cz
    bbox_msg.pose.orientation.w = 1
    bbox_msg.dimensions.x = xsize
    bbox_msg.dimensions.y = ysize
    bbox_msg.dimensions.z = zsize
    return bbox_msg


def extent_array_to_bbox(extent_3d):
    min_x = extent_3d[0, 0]
    max_x = extent_3d[0, 1]
    min_y = extent_3d[1, 0]
    max_y = extent_3d[1, 1]
    min_z = extent_3d[2, 0]
    max_z = extent_3d[2, 1]
    xsize = abs(max_x - min_x)
    ysize = abs(max_y - min_y)
    zsize = abs(max_z - min_z)

    cx = (max_x + min_x) / 2
    cy = (max_y + min_y) / 2
    cz = (max_z + min_z) / 2

    bbox_msg = BoundingBox()
    bbox_msg.pose.position.x = cx
    bbox_msg.pose.position.y = cy
    bbox_msg.pose.position.z = cz
    bbox_msg.pose.orientation.w = 1
    bbox_msg.dimensions.x = xsize
    bbox_msg.dimensions.y = ysize
    bbox_msg.dimensions.z = zsize
    return bbox_msg


def extent_to_bbox(extent_3d):
    min_x, max_x, min_y, max_y, min_z, max_z = extent_3d
    xsize = abs(max_x - min_x)
    ysize = abs(max_y - min_y)
    zsize = abs(max_z - min_z)

    cx = (max_x + min_x) / 2
    cy = (max_y + min_y) / 2
    cz = (max_z + min_z) / 2

    bbox_msg = BoundingBox()
    bbox_msg.pose.position.x = cx
    bbox_msg.pose.position.y = cy
    bbox_msg.pose.position.z = cz
    bbox_msg.pose.orientation.w = 1
    bbox_msg.dimensions.x = xsize
    bbox_msg.dimensions.y = ysize
    bbox_msg.dimensions.z = zsize
    return bbox_msg


def viz_action_sample_bbox(gripper_bbox_pub: rospy.Publisher, gripper_extent):
    gripper_bbox_msg = extent_array_to_bbox(gripper_extent)
    gripper_bbox_msg.header.frame_id = 'world'
    gripper_bbox_pub.publish(gripper_bbox_msg)
