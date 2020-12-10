import pathlib

import rosbag
import roslaunch
import rospy
from gazebo_msgs.srv import SetPhysicsPropertiesRequest, GetPhysicsPropertiesRequest, SetLinkState, SetLinkStateRequest
from link_bot_pycommon.base_services import BaseServices
from std_srvs.srv import EmptyRequest, Empty


class GazeboServices(BaseServices):

    def __init__(self):
        super().__init__()
        self.max_step_size = None
        self.gazebo_process = None

        # Yes, absolute paths here are what I want. I don't want these namespaced by the robot
        self.set_link_state = self.add_required_service('/gazebo/set_link_state', SetLinkState)
        self.pause_srv = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.play_srv = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

    def restore_from_bag(self, bagfile_name: pathlib.Path):
        # run a few times to really make sure it happens
        for _ in range(4):
            with rosbag.Bag(bagfile_name) as bag:
                saved_links_states = next(iter(bag.read_messages()))[1]
                set_link_state_req = SetLinkStateRequest()
                set_link_state_req.link_state = saved_links_states

                n = len(saved_links_states.name)
                for i in range(n):
                    name = saved_links_states.name[i]
                    pose = saved_links_states.pose[i]
                    twist = saved_links_states.twist[i]
                    set_req = SetLinkStateRequest()
                    set_req.link_state.link_name = name
                    set_req.link_state.pose = pose
                    set_req.link_state.twist = twist
                    self.set_link_state(set_req)

    def launch(self, params, **kwargs):
        gui = kwargs.get("gui", True)
        world = kwargs.get("world", None)
        launch_file_name = params['launch']
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        roslaunch_args = ['link_bot_gazebo', launch_file_name, f"gui:={str(gui).lower()}"]
        if world:
            roslaunch_args.append(f"world:={world}")

        roslaunch_file = roslaunch.rlutil.resolve_launch_arguments(roslaunch_args)[0]
        roslaunch_args = roslaunch_args[2:]

        launch_info = [(roslaunch_file, roslaunch_args)]

        self.gazebo_process = roslaunch.parent.ROSLaunchParent(uuid, launch_info)
        self.gazebo_process.start()

        # wait until services are available before returning
        self.wait_for_services()

        self.play()

    def kill(self):
        self.gazebo_process.shutdown()

    def setup_env(self, verbose: int, real_time_rate: float, max_step_size: float, play: bool = True):
        # set up physics
        get_physics_msg = GetPhysicsPropertiesRequest()
        current_physics = self.get_physics.call(get_physics_msg)
        set_physics_msg = SetPhysicsPropertiesRequest()
        set_physics_msg.gravity = current_physics.gravity
        set_physics_msg.ode_config = current_physics.ode_config
        set_physics_msg.max_update_rate = real_time_rate / max_step_size
        if max_step_size is None:
            max_step_size = current_physics.time_step
        self.max_step_size = max_step_size
        set_physics_msg.time_step = max_step_size
        self.set_physics.call(set_physics_msg)

        if play:
            self.play()

    def play(self):
        try:
            self.play_srv(EmptyRequest())
        except rospy.ServiceException:
            pass

    def pause(self):
        try:
            self.pause_srv(EmptyRequest())
        except rospy.ServiceException:
            pass


def gz_scope(*args):
    return "::".join(args)
