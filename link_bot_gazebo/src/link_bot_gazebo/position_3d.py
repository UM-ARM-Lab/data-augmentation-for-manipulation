import rospy
from peter_msgs.srv import *


class Position3D:

    def __init__(self):
        self.register_controller_srv = rospy.ServiceProxy("/position_3d_plugin/register", RegisterPosition3DController)
        self.follow_srv = rospy.ServiceProxy("/position_3d_plugin/follow", Position3DFollow)
        self.enable_srv = rospy.ServiceProxy("/position_3d_plugin/enable", Position3DEnable)
        self.set_srv = rospy.ServiceProxy("/position_3d_plugin/set", Position3DAction)
        self.move_srv = rospy.ServiceProxy("/position_3d_plugin/move", Position3DAction)
        self.wait_srv = rospy.ServiceProxy("/position_3d_plugin/wait", Position3DWait)
        self.get_srv = rospy.ServiceProxy("/position_3d_plugin/get", GetPosition3D)
        self.list_srv = rospy.ServiceProxy("/position_3d_plugin/list", Position3DList)

    def list(self) -> Position3DListResponse:
        return self.list_srv(Position3DListRequest())

    def register(self, msg: RegisterPosition3DControllerRequest) -> RegisterPosition3DControllerResponse:
        return self.register_controller_srv(msg)

    def follow(self, msg: Position3DFollowRequest) -> Position3DFollowResponse:
        return self.follow_srv(msg)

    def enable(self, msg: Position3DEnableRequest) -> Position3DEnableResponse:
        return self.enable_srv(msg)

    def set(self, msg: Position3DActionRequest) -> Position3DActionResponse:
        return self.set_srv(msg)

    def move(self, msg: Position3DActionRequest) -> Position3DActionResponse:
        return self.move_srv(msg)

    def wait(self, msg: Position3DWaitRequest) -> Position3DWaitResponse:
        return self.wait_srv(msg)

    def get(self, scoped_link_name: str) -> GetPosition3DResponse:
        return self.get_srv(GetPosition3DRequest(scoped_link_name=scoped_link_name))

    def exists(self, name: str):
        res = self.list()
        return name in res.controller_names
