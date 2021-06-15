from visualization_msgs.msg import Marker, MarkerArray


def make_delete_marker(marker_id: int, ns: str):
    m = Marker(action=Marker.DELETEALL, ns=ns, id=marker_id)
    return m


def make_delete_markerarray(marker_id: int, ns: str):
    m = Marker(action=Marker.DELETEALL, ns=ns, id=marker_id)
    msg = MarkerArray(markers=[m])
    return msg