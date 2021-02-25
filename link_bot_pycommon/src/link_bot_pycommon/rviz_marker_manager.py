from typing import Dict, Tuple

import rospy
from link_bot_data.visualization import make_delete_marker


class RVizMarkerManager:

    def __init__(self):
        self.markers: Dict[str, Dict[int, Tuple[rospy.Publisher, str]]] = {}

    def add(self, label, index):
        pass

    def delete(self, label: str):
        if label in self.markers:
            for marker_id, (publisher, ns) in self.markers[label].items():
                m = make_delete_marker(ns=ns, marker_id=marker_id)
                publisher.publish(m)

    def delete_all(self):
        pass
