import zlib
from typing import Dict, List, Optional

from arm_video_recorder.srv import TriggerVideoRecording, TriggerVideoRecordingRequest

import rospy
from arm_gazebo_msgs.srv import ComputeOccupancy, GetWorldInitialSDFResponse, GetWorldInitialSDF
from gazebo_msgs.srv import GetPhysicsProperties, SetPhysicsProperties
from geometry_msgs.msg import Pose
from link_bot_gazebo import gazebo_utils
from peter_msgs.srv import WorldControl, WorldControlRequest


class BaseServices:

    def __init__(self):
        self.service_names = []

        gazebo_utils.resume()

        self.world_control = self.add_required_service('arm_gazebo/world_control', WorldControl)
        self.get_physics = self.add_required_service('/gazebo/get_physics_properties', GetPhysicsProperties)
        self.set_physics = self.add_required_service('/gazebo/set_physics_properties', SetPhysicsProperties)

        # services we don't absolutly want to wait for on startup
        self.compute_occupancy = rospy.ServiceProxy('/occupancy', ComputeOccupancy)
        self.record = rospy.ServiceProxy('video_recorder', TriggerVideoRecording)

    def wait_for_services(self):
        for service_name in self.service_names:
            rospy.wait_for_service(service_name, timeout=10)

    def launch(self, params, **kwargs):
        pass

    def kill(self):
        pass

    def move_objects(self, object_moves: Dict[str, Pose]):
        pass

    def start_record_trial(self, filename):
        print("Starting recording!")
        start_msg = TriggerVideoRecordingRequest()
        start_msg.record = True
        start_msg.filename = filename
        start_msg.timeout_in_sec = 600.0
        try:
            self.record(start_msg)
        except rospy.ServiceException:
            rospy.logerr("Recording failed!")

    def stop_record_trial(self):
        print("Stopping recording!")
        stop_msg = TriggerVideoRecordingRequest()
        stop_msg.record = False
        try:
            self.record(stop_msg)
        except rospy.ServiceException:
            rospy.logerr("Recording failed!")

    def setup_env(self, verbose: int, real_time_rate: float, max_step_size: float):
        raise NotImplementedError()

    def add_required_service(self, service_name, service_type):
        self.service_names.append(service_name)
        return rospy.ServiceProxy(service_name, service_type)

    def restore_from_bag(self, bagfile_name, excluded_models: Optional[List[str]] = None):
        pass

    def step(self, steps: int = 1):
        step = WorldControlRequest()
        step.steps = steps
        self.world_control(step)

    def pause(self):
        pass

    def play(self):
        pass
