from arc_utilities.listener import Listener
from arc_utilities.tf2wrapper import TF2Wrapper
from link_bot_pycommon.ros_pycommon import transform_points_to_robot_frame
from sensor_msgs.msg import PointCloud2


class GetCdcpdState:

    def __init__(self, tf: TF2Wrapper, root_link: str, key: str = 'rope'):
        self.tf = tf
        self.key = key
        self.cdcpd_listener = Listener("cdcpd/output", PointCloud2)
        self.root_link = root_link

    def get_state(self):
        cdcpd_msg: PointCloud2 = self.cdcpd_listener.get()

        points = transform_points_to_robot_frame(self.tf, cdcpd_msg, robot_frame_id=self.root_link)

        cdcpd_vector = points.flatten()

        return {
            self.key: cdcpd_vector,
        }
