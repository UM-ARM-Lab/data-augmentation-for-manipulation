from link_bot_pycommon.grid_utils_np import extent_to_env_size, extent_to_center
from visualization_msgs.msg import Marker


def make_box_marker_from_extents(extent):
    m = Marker()
    ysize, xsize, zsize = extent_to_env_size(extent)
    xcenter, ycenter, zcenter = extent_to_center(extent)
    m.scale.x = xsize
    m.scale.y = ysize
    m.scale.z = zsize
    m.action = Marker.ADD
    m.type = Marker.CUBE
    m.pose.position.x = xcenter
    m.pose.position.y = ycenter
    m.pose.position.z = zcenter
    m.pose.orientation.w = 1
    return m