from typing import Dict, Optional

import numpy as np
from dm_control import composer
from dm_envs.blocks_env import my_blocks
from matplotlib import colors

import ros_numpy
import rospy
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from sensor_msgs.msg import Image
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker


class BlocksScenario(ScenarioWithVisualization):
    """
    the state here is defined by the dm_control Env, "my_blocks"
    """

    def __init__(self):
        ScenarioWithVisualization.__init__(self)
        self.task = None
        self.env = None
        self.action_spec = None
        self.num_blocks = None

        self.camera_pub = rospy.Publisher("camera", Image, queue_size=10)

        self.num_blocks = 20
        self.task = my_blocks(num_blocks=self.num_blocks)
        # we don't want episode termination to be decided by dm_control, we do that ourselves elsewhere
        self.env = composer.Environment(self.task, time_limit=9999, random_state=0)
        self.action_spec = self.env.action_spec()
        print("constructing BlockScenario")

    def on_before_data_collection(self, params: Dict):
        # modified the input dict!
        s = self.get_state()
        params['state_keys'] = list(s.keys())
        params['env_keys'] = []
        params['action_keys'] = ['mjaction']
        params['state_metadata_keys'] = []

    def get_environment(self, params: Dict, **kwargs):
        # not the mujoco "env", this means the static obstacles and workspaces geometry
        return {}

    def get_state(self):
        state = self.env._observation_updater.get_observation()
        return state

    def plot_environment_rviz(self, environment: Dict, **kwargs):
        pass

    def plot_state_rviz(self, state: Dict, **kwargs):
        ns = kwargs.get("label", "")
        color_msg = ColorRGBA(*colors.to_rgba(kwargs.get("color", "r")))
        if 'a' in kwargs:
            color_msg.a = kwargs['a']
            a = kwargs['a']
        else:
            a = 1.0
        idx = kwargs.get("idx", 0)

        # TODO: plot the robot arm state

        msg = MarkerArray()
        for i in range(self.num_blocks):
            box_position = state[f'box{i}/position']
            box_orientation = state[f'box{i}/orientation']

            block_marker = Marker()
            block_marker.header.frame_id = 'world'
            block_marker.action = Marker.ADD
            block_marker.type = Marker.CUBE
            block_marker.id = idx * self.num_blocks + i
            block_marker.ns = ns
            block_marker.color = color_msg
            block_marker.pose.position.x = box_position[0, 0]
            block_marker.pose.position.y = box_position[0, 1]
            block_marker.pose.position.z = box_position[0, 2]
            block_marker.pose.orientation.w = box_orientation[0, 0]
            block_marker.pose.orientation.x = box_orientation[0, 1]
            block_marker.pose.orientation.y = box_orientation[0, 2]
            block_marker.pose.orientation.z = box_orientation[0, 3]
            block_marker.scale.x = self.task.box_length
            block_marker.scale.y = self.task.box_length
            block_marker.scale.z = self.task.box_length
            msg.markers.append(block_marker)

        self.state_viz_pub.publish(msg)

        img = state['front_close'][0]
        img_msg = ros_numpy.msgify(Image, img, encoding='rgb8')
        img_msg.header.frame_id = 'world'
        self.camera_pub.publish(img_msg)

    def plot_action_rviz(self, state: Dict, action: Dict, **kwargs):
        pass

    def sample_action(self,
                      action_rng: np.random.RandomState,
                      environment: Dict,
                      state: Dict,
                      action_params: Dict,
                      validate, stateless: Optional[bool] = False):
        action = np.random.uniform(self.action_spec.minimum, self.action_spec.maximum, self.action_spec.shape)
        action_dict = {
            'mjaction': action,
        }
        return action_dict, False

    def execute_action(self, environment, state, action: Dict):
        self.env.step(action['mjaction'])
        end_trial = False
        return end_trial

    def needs_reset(self, state: Dict, params: Dict):
        return False

    def __repr__(self):
        return "blocks"
