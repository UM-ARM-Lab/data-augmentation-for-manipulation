from typing import Dict, Tuple

import rospy
from link_bot_data.visualization import make_delete_marker


class RVizMarkerManager:

    def __init__(self):
        self.markers: Dict[str, Dict[int, Tuple[rospy.Publisher, str]]] = {}

    def add(self, publisher: rospy.Publisher, label: str, index: int):
        if publisher.name not in self.markers:
            self.markers[publisher.name] = {}

    def delete(self, publisher: rospy.Publisher, label: str, index: int):
        m = make_delete_marker(ns=label, marker_id=index)
        publisher.publish(m)

    def delete_all(self):
        pass
