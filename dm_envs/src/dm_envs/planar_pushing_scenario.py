from typing import Dict, Optional

import numpy as np
import tensorflow as tf
from dm_control import composer
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d

import ros_numpy
import rospy
from jsk_recognition_msgs.msg import BoundingBox
from link_bot_data.visualization_common import make_delete_markerarray
from link_bot_pycommon.bbox_visualization import viz_action_sample_bbox
from link_bot_pycommon.experiment_scenario import get_action_sample_extent, is_out_of_bounds
from link_bot_pycommon.grid_utils import extent_to_env_shape
from link_bot_pycommon.pycommon import yaw_diff
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from sdf_tools.utils_3d import compute_sdf_and_gradient
from sensor_msgs.msg import Image, JointState
from visualization_msgs.msg import MarkerArray

ARM_NAME = 'jaco_arm'
HAND_NAME = 'primitive_hand'


def get_joint_position(state):
    sin_cos = state[f'{ARM_NAME}/joints_pos']
    sin = sin_cos[0, :, 0]
    cos = sin_cos[0, :, 1]
    angles = np.arctan2(sin, cos)
    return angles


def get_joint_velocities(state):
    return state[f'{ARM_NAME}/joints_vel'][0]


def get_tcp_pos(state):
    return state[f'{ARM_NAME}/{HAND_NAME}/tcp_pos'][0]


def sample_delta_xy(action_params: Dict, action_rng: np.random.RandomState):
    d = action_params['max_distance_gripper_can_move']
    dx = action_rng.uniform(-d, d)
    dy = action_rng.uniform(-d, d)
    z_noise_max = 0.01
    z_noise = action_rng.uniform(-z_noise_max, z_noise_max)
    return [dx, dy, z_noise]


def transformation_matrices_from_pos_quat(positions, quats):
    """

    Args:
        positions: [b, m, T, 3]
        quats: [b, m, T, 4]

    Returns:  [b, m, T, 4, 4]

    """
    rmat = rotation_matrix_3d.from_quaternion(quats)  # [b,m,T,3,3]
    bottom_row = tf.constant([0, 0, 0, 1], dtype=tf.float32)  # [4]
    mat34 = tf.concat([rmat, positions[..., None]], axis=-1)
    bottom_row = tf.expand_dims(tf.ones_like(quats, tf.float32), axis=-2) * bottom_row[None, None, None, None]
    mat44 = tf.concat([mat34, bottom_row], axis=-2)
    return mat44


def wxyz2xyzw(quat):
    w, x, y, z = tf.unstack(quat, axis=-1)
    return tf.stack([x, y, z, w], axis=-1)


class PlanarPushingScenario(ScenarioWithVisualization):

    def __init__(self):
        ScenarioWithVisualization.__init__(self)
        self.task = None
        self.env = None
        self.action_spec = None

        self.camera_pub = rospy.Publisher("camera", Image, queue_size=10)
        self.gripper_bbox_pub = rospy.Publisher('gripper_bbox_pub', BoundingBox, queue_size=10, latch=True)
        self.joint_states_pub = rospy.Publisher('jaco_arm/joint_states', JointState, queue_size=10)
        self.viz_aug_pub = rospy.Publisher('viz_aug', MarkerArray, queue_size=10)

        self.last_action = None
        self.max_action_attempts = 100

    def on_before_data_collection(self, params: Dict):
        self.task = self.make_dm_task(params)
        # we don't want episode termination to be decided by dm_control, we do that ourselves elsewhere
        self.env = composer.Environment(self.task, time_limit=9999, random_state=0)
        self.env.reset()
        self.action_spec = self.env.action_spec()

        # modified the input dict!
        s = self.get_state()

        params['state_keys'] = list(s.keys())
        params['env_keys'] = [
            'res',
            'extent',
            'env',
            'origin_point',
            'sdf',
            'sdf_grad',
        ]
        params['action_keys'] = ['gripper_position']
        params['state_metadata_keys'] = []
        params['gripper_keys'] = ['jaco_arm/primitive_hand/tcp_pos', 'jaco_arm/primitive_hand/orientation']
        params['augmentable_state_keys'] = []

        params['points_state_keys'] = list(filter(self.is_points_key, s.keys()))

    def get_environment(self, params: Dict, **kwargs):
        # not the mujoco "env", this means the static obstacles and workspaces geometry
        res = np.float32(0.005)
        extent = np.array(params['extent'])
        origin_point = extent[[0, 2, 4]]
        shape = extent_to_env_shape(extent, res)
        mock_floor_voxel_grid = np.zeros(shape, np.float32)
        mock_floor_voxel_grid[:, :, 0] = 1.0

        sdf, sdf_grad = compute_sdf_and_gradient(mock_floor_voxel_grid, res, origin_point)

        return {
            'res':          res,
            'extent':       extent,
            'env':          mock_floor_voxel_grid,
            'origin_point': origin_point,
            'sdf':          sdf,
            'sdf_grad':     sdf_grad,
        }

    def get_state(self):
        state = self.env._observation_updater.get_observation()
        joint_names = [n.replace(f'{ARM_NAME}/', '') for n in self.task.joint_names]
        state['joint_names'] = np.array(joint_names)

        return state

    def plot_state_rviz(self, state: Dict, **kwargs):
        if 'jaco_arm/joints_pos' in state:
            joint_state = self.get_joint_state_msg(state)
            self.joint_states_pub.publish(joint_state)

        if 'front_close' in state:
            img = state['front_close'][0]
            img_msg = ros_numpy.msgify(Image, img, encoding='rgb8')
            img_msg.header.frame_id = 'world'
            self.camera_pub.publish(img_msg)

    def get_joint_state_msg(self, state):
        joint_position = get_joint_position(state).tolist()
        return self.joint_position_to_msg(joint_position, state['joint_names'])

    def joint_position_to_msg(self, joint_position, joint_names):
        joint_state = JointState()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.position = joint_position
        joint_state.name = [n.replace(f'{ARM_NAME}/', '') for n in joint_names]
        return joint_state

    def plot_action_rviz(self, state: Dict, action: Dict, **kwargs):
        pass

    def sample_action(self,
                      action_rng: np.random.RandomState,
                      environment: Dict,
                      state: Dict,
                      action_params: Dict,
                      validate, stateless: Optional[bool] = False):
        viz_action_sample_bbox(self.gripper_bbox_pub, get_action_sample_extent(action_params))

        start_gripper_position = get_tcp_pos(state)
        action_dict = {
            'gripper_position': start_gripper_position,
        }

        # first check if any objects are wayyy to far
        num_objs = state['num_objs'][0]
        for i in range(num_objs):
            obj_position = state[f'obj{i}/position'][0]
            out_of_bounds = is_out_of_bounds(obj_position, action_params['extent'])
            if out_of_bounds:
                return action_dict, (invalid := False)  # this will cause the current trajectory to be thrown out

        for _ in range(self.max_action_attempts):
            repeat_probability = action_params['repeat_delta_gripper_motion_probability']
            if self.last_action is not None and action_rng.uniform(0, 1) < repeat_probability:
                gripper_delta_position = self.last_action['gripper_delta_position']
            else:
                gripper_delta_position = sample_delta_xy(action_params, action_rng)

            gripper_position = start_gripper_position + gripper_delta_position
            gripper_position[2] = 0.01

            self.tf.send_transform(gripper_position, [0, 0, 0, 1], 'world', 'sample_action_gripper_position')

            out_of_bounds = is_out_of_bounds(gripper_position, action_params['gripper_action_sample_extent'])
            if out_of_bounds and validate:
                self.last_action = None
                continue

            action = {
                'gripper_position':       gripper_position,
                'gripper_delta_position': gripper_delta_position,
            }

            self.last_action = action
            return action, (invalid := False)

        rospy.logwarn("Could not find a valid action, executing an invalid one")
        return action_dict, (invalid := False)

    def execute_action(self, environment, state, action: Dict):
        target_cartesian_position = action['gripper_position']

        # we picked a new end effector pose, now solve IK to turn that into a joint configuration
        success, target_joint_position = self.task.solve_position_ik(self.env.physics, target_cartesian_position)
        if not success:
            rospy.logwarn("failed to solve IK! continuing anyways")

        current_position = get_joint_position(state)
        kP = 10.0

        max_substeps = 50
        for substeps in range(max_substeps):
            # p-control to achieve joint positions using the lower level velocity controller
            velocity_cmd = yaw_diff(target_joint_position, current_position) * kP
            self.env.step(velocity_cmd)
            state = self.get_state()
            # self.plot_state_rviz(state)

            current_position = get_joint_position(state)
            max_error = max(yaw_diff(target_joint_position, current_position))
            max_vel = max(abs(get_joint_velocities(state)))
            reached = max_error < 0.01
            stopped = max_vel < 0.002
            if reached and stopped:
                break

        return (end_trial := False)

    def needs_reset(self, state: Dict, params: Dict):
        return False

    def randomize_environment(self, env_rng: np.random.RandomState, params: Dict):
        self.env.reset()

    def reset_viz(self):
        super().reset_viz()
        m = make_delete_markerarray(ns='viz_aug')
        self.viz_aug_pub.publish(m)

    def make_dm_task(self, params):
        raise NotImplementedError()

    @staticmethod
    def is_points_key(k):
        raise NotImplementedError()
