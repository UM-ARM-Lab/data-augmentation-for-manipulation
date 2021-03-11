import rospy
from gazebo_msgs.srv import GetModelState, GetModelStateResponse, GetModelStateRequest


def get_gazebo_kinect_pose(model_name="kinect2"):
    get_srv = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
    res: GetModelStateResponse = get_srv(GetModelStateRequest(model_name=model_name))
    return res.pose
