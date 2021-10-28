from typing import Dict, Optional

import numpy as np
from dm_control import composer
from matplotlib import colors

import ros_numpy
import rospy
from dm_envs.blocks_env import my_blocks
from jsk_recognition_msgs.msg import BoundingBox
from link_bot_pycommon.bbox_visualization import viz_action_sample_bbox
from link_bot_pycommon.experiment_scenario import get_action_sample_extent, is_out_of_bounds, sample_delta_position
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker

ARM_NAME = 'jaco_arm'
HAND_NAME = 'primitive_hand'


class BlocksScenario(ScenarioWithVisualization):
    """
    the state here is defined by the dm_control Env, "my_blocks"
    """

    def __init__(self):
        ScenarioWithVisualization.__init__(self)
        self.task = None
        self.env = None
        self.action_spec = None

        self.camera_pub = rospy.Publisher("camera", Image, queue_size=10)
        self.gripper_bbox_pub = rospy.Publisher('gripper_bbox_pub', BoundingBox, queue_size=10, latch=True)
        self.kinova_joint_states_pub = rospy.Publisher('kinova/joint_states', JointState, queue_size=10)

        self.last_action = None
        self.max_action_attempts = 100

    def on_before_data_collection(self, params: Dict):
        self.task = my_blocks(params)
        # we don't want episode termination to be decided by dm_control, we do that ourselves elsewhere
        self.env = composer.Environment(self.task, time_limit=9999, random_state=0)
        self.env.reset()
        self.action_spec = self.env.action_spec()
        print("constructing BlockScenario")

        # modified the input dict!
        s = self.get_state()
        params['state_keys'] = list(s.keys())
        params['env_keys'] = []
        params['action_keys'] = ['joint_position']
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
        joint_state = JointState()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.position = state[f'{ARM_NAME}/joints_pos'][0, :, 0].tolist()
        joint_state.name = [n.replace(f'{ARM_NAME}/', '') for n in self.task.joint_names]
        self.kinova_joint_states_pub.publish(joint_state)

        num_blocks = state['num_blocks'][0]
        msg = MarkerArray()
        for i in range(num_blocks):
            box_position = state[f'box{i}/position']
            box_orientation = state[f'box{i}/orientation']

            block_marker = Marker()
            block_marker.header.frame_id = 'world'
            block_marker.action = Marker.ADD
            block_marker.type = Marker.CUBE
            block_marker.id = idx * num_blocks + i
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
        viz_action_sample_bbox(self.gripper_bbox_pub, get_action_sample_extent(action_params))

        # sample random delta in x and y

        for _ in range(self.max_action_attempts):
            repeat_probability = action_params['repeat_delta_gripper_motion_probability']
            if self.last_action is not None and action_rng.uniform(0, 1) < repeat_probability:
                gripper_delta_position = self.last_action['gripper_delta_position']
            else:
                gripper_delta_position = sample_delta_position(action_params, action_rng)

            gripper_position = state[f'{ARM_NAME}/{HAND_NAME}/tcp_pos'][0] + gripper_delta_position

            out_of_bounds = is_out_of_bounds(gripper_position, action_params['gripper_action_sample_extent'])
            if out_of_bounds and validate:
                continue

            # we picked a new end effector pose, now solve IK to turn that into a joint configuration
            success, joint_position = self.task.solve_position_ik(self.env.physics, gripper_position)
            if not success:
                continue

            action = {
                'joint_position':         joint_position,
                'gripper_delta_position': gripper_delta_position,
            }

            self.last_action = action
            return action, (invalid := False)

        print("WHY IS THIS 3D???")
        current_joint_position = state[f'{ARM_NAME}/joints_pos'][0, :, 0]
        action_dict = {
            'joint_position': current_joint_position,
        }
        rospy.logwarn("Could not find a valid action, executing an invalid one")
        return action_dict, (invalid := False)

    def execute_action(self, environment, state, action: Dict):
        self.env.step(action['joint_position'])
        end_trial = False
        return end_trial

    def needs_reset(self, state: Dict, params: Dict):
        return False

    def __repr__(self):
        return "blocks"
