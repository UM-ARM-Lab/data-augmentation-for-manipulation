import re
from typing import Dict, Optional

import numpy as np
import tensorflow as tf
from dm_control import composer
from matplotlib import colors
from pyjacobian_follower import IkParams
from tensorflow_graphics.geometry.transformation import quaternion
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d

import ros_numpy
import rospy
from dm_envs.blocks_env import my_blocks
from jsk_recognition_msgs.msg import BoundingBox
from link_bot_data.visualization_common import make_delete_markerarray
from link_bot_pycommon.bbox_visualization import viz_action_sample_bbox
from link_bot_pycommon.experiment_scenario import get_action_sample_extent, is_out_of_bounds
from link_bot_pycommon.grid_utils import extent_to_env_shape
from link_bot_pycommon.pycommon import yaw_diff
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from moonshine.geometry import transform_points_3d
from moonshine.moonshine_utils import repeat_tensor
from sdf_tools.utils_3d import compute_sdf_and_gradient
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker

ARM_NAME = 'jaco_arm'
HAND_NAME = 'primitive_hand'


def get_joint_position(state):
    sin = state[f'{ARM_NAME}/joints_pos'][0, :, 0]
    cos = state[f'{ARM_NAME}/joints_pos'][0, :, 1]
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


def blocks_to_points(positions, quats, size):
    """

    Args:
        positions:  [b, m, T, 3]
        quats:  [b, m, T, 4]
        size:  [b, T]

    Returns: [b, m, T, 8, 3]

    """
    m = positions.shape[1]
    sized_cube_points = size_to_points(size)  # [b, T, 8, 3]
    sized_cubes_points = repeat_tensor(sized_cube_points, m, axis=1, new_axis=True)  # [b, m, T, 8, 3]
    transform_matrix = transformation_matrices_from_pos_quat(positions, quats)  # [b, m, T, 4, 4]
    transform_matrix = repeat_tensor(transform_matrix, 8, 3, True)
    obj_points = transform_points_3d(transform_matrix, sized_cubes_points)  # [b, m, T, 8, 3]
    return obj_points


def size_to_points(size):
    """

    Args:
        size: [b, T] where each element is the size of the cube. it can vary of batch/time

    Returns: [b, T, 8, 3] where 8 represents the corners the cube, and 3 is x,y,z.

    """
    unit_cube_points = tf.constant([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
                                    [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]], tf.float32) - 0.5
    return unit_cube_points[None, None] * size[:, :, None, None]  # [b, T, 8, 3]


def wxyz2xyzw(quat):
    w, x, y, z = tf.unstack(quat, axis=-1)
    return tf.stack([x, y, z, w], axis=-1)


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
        self.joint_states_pub = rospy.Publisher('jaco_arm/joint_states', JointState, queue_size=10)
        self.blocks_aug_pub = rospy.Publisher('blocks_aug', MarkerArray, queue_size=10)

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
        self.plot_state_rviz(s)

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
        params['augmentable_state_keys'] = [k for k in s.keys() if 'block' in k]

        def _is_points_key(k):
            return any([
                re.match('block.*position', k),
                k == 'jaco_arm/primitive_hand/tcp_pos',
            ])

        params['points_state_keys'] = list(filter(_is_points_key, s.keys()))

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
        ns = kwargs.get("label", "")
        color_msg = ColorRGBA(*colors.to_rgba(kwargs.get("color", "r")))
        if 'a' in kwargs:
            color_msg.a = kwargs['a']
            a = kwargs['a']
        else:
            a = 1.0
        idx = kwargs.get("idx", 0)

        joint_state = self.get_joint_state_msg(state)
        self.joint_states_pub.publish(joint_state)

        num_blocks = state['num_blocks'][0]
        block_size = state['block_size'][0]
        msg = MarkerArray()
        for i in range(num_blocks):
            block_position = state[f'block{i}/position']
            block_orientation = state[f'block{i}/orientation']

            block_marker = Marker()
            block_marker.header.frame_id = 'world'
            block_marker.action = Marker.ADD
            block_marker.type = Marker.CUBE
            block_marker.id = idx * num_blocks + i
            block_marker.ns = ns
            block_marker.color = color_msg
            block_marker.pose.position.x = block_position[0, 0]
            block_marker.pose.position.y = block_position[0, 1]
            block_marker.pose.position.z = block_position[0, 2]
            block_marker.pose.orientation.w = block_orientation[0, 0]
            block_marker.pose.orientation.x = block_orientation[0, 1]
            block_marker.pose.orientation.y = block_orientation[0, 2]
            block_marker.pose.orientation.z = block_orientation[0, 3]
            block_marker.scale.x = block_size
            block_marker.scale.y = block_size
            block_marker.scale.z = block_size
            msg.markers.append(block_marker)

        self.state_viz_pub.publish(msg)

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

        for _ in range(self.max_action_attempts):
            repeat_probability = action_params['repeat_delta_gripper_motion_probability']
            if self.last_action is not None and action_rng.uniform(0, 1) < repeat_probability:
                gripper_delta_position = self.last_action['gripper_delta_position']
            else:
                gripper_delta_position = sample_delta_xy(action_params, action_rng)

            gripper_position = start_gripper_position + gripper_delta_position

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

        action_dict = {
            'gripper_position': start_gripper_position,
        }
        rospy.logwarn("Could not find a valid action, executing an invalid one")
        return action_dict, (invalid := False)

    def execute_action(self, environment, state, action: Dict):
        target_cartesian_position = action['gripper_position']

        # we picked a new end effector pose, now solve IK to turn that into a joint configuration
        success, target_joint_position = self.task.solve_position_ik(self.env.physics, target_cartesian_position)
        if not success:
            print("failed to solve IK! continuing anyways")

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

    def compute_obj_points(self, inputs: Dict, num_object_interp: int, batch_size: int):
        """

        Args:
            inputs: contains the poses and size of the blocks, over a whole trajectory, which we convert into points
            num_object_interp:
            batch_size:

        Returns:

        """
        size = inputs['block_size'][:, :, 0]  # [b, T]
        num_blocks = inputs['num_blocks'][0, 0, 0]  # assumed fixed across batch/time
        positions = []  # [b, m, T, 3]
        quats = []  # [b, m, T, 4]
        for block_idx in range(num_blocks):
            pos = inputs[f"block{block_idx}/position"][:, :, 0]  # [b, T, 3]
            # in our mujoco dataset the quaternions are stored w,x,y,z but the rest of our code assumes xyzw
            quat = inputs[f"block{block_idx}/orientation"][:, :, 0]  # [b, T, 4]
            quat = wxyz2xyzw(quat)
            positions.append(pos)
            quats.append(quat)
        positions = tf.stack(positions, axis=1)
        quats = tf.stack(quats, axis=1)

        obj_points = blocks_to_points(positions, quats, size)

        # combine to get one set of points per object
        obj_points = tf.reshape(obj_points, [obj_points.shape[0], obj_points.shape[1], -1, 3])

        return obj_points

    def apply_object_augmentation_no_ik(self,
                                        m,
                                        to_local_frame,
                                        inputs: Dict,
                                        batch_size,
                                        time,
                                        h: int,
                                        w: int,
                                        c: int,
                                        ):
        """

        Args:
            m: [b, 4, 4]
            to_local_frame: [b, 1, 3]  the 1 can also be equal to time
            inputs:
            batch_size:
            time:
            h:
            w:
            c:

        Returns:

        """
        # apply those to the rope and grippers
        rope_points = tf.reshape(inputs[add_predicted('rope')], [batch_size, time, -1, 3])
        left_gripper_point = inputs[add_predicted('left_gripper')]
        right_gripper_point = inputs[add_predicted('right_gripper')]
        left_gripper_points = tf.expand_dims(left_gripper_point, axis=-2)
        right_gripper_points = tf.expand_dims(right_gripper_point, axis=-2)

        def _transform(m, points, _to_local_frame):
            points_local_frame = points - _to_local_frame
            points_local_frame_aug = transform_points_3d(m, points_local_frame)
            return points_local_frame_aug + _to_local_frame

        # m is expanded to broadcast across batch & num_points dimensions
        rope_points_aug = _transform(m[:, None, None], rope_points, to_local_frame[:, None])
        left_gripper_points_aug = _transform(m[:, None, None], left_gripper_points, to_local_frame[:, None])
        right_gripper_points_aug = _transform(m[:, None, None], right_gripper_points, to_local_frame[:, None])

        # compute the new action
        left_gripper_position = inputs['left_gripper_position']
        right_gripper_position = inputs['right_gripper_position']
        # m is expanded to broadcast across batch dimensions
        left_gripper_position_aug = _transform(m[:, None], left_gripper_position, to_local_frame)
        right_gripper_position_aug = _transform(m[:, None], right_gripper_position, to_local_frame)

        rope_aug = tf.reshape(rope_points_aug, [batch_size, time, -1])
        left_gripper_aug = tf.reshape(left_gripper_points_aug, [batch_size, time, -1])
        right_gripper_aug = tf.reshape(right_gripper_points_aug, [batch_size, time, -1])

        # Now that we've updated the state/action in inputs, compute the local origin point
        state_aug_0 = {
            'left_gripper':  left_gripper_aug[:, 0],
            'right_gripper': right_gripper_aug[:, 0],
            'rope':          rope_aug[:, 0]
        }
        local_center_aug = self.local_environment_center_differentiable(state_aug_0)
        res = inputs['res']
        local_origin_point_aug = batch_center_res_shape_to_origin_point(local_center_aug, res, h, w, c)

        object_aug_update = {
            add_predicted('rope'):          rope_aug,
            add_predicted('left_gripper'):  left_gripper_aug,
            add_predicted('right_gripper'): right_gripper_aug,
            'left_gripper_position':        left_gripper_position_aug,
            'right_gripper_position':       right_gripper_position_aug,
        }

        if DEBUG_VIZ_STATE_AUG:
            stepper = RvizSimpleStepper()
            for b in debug_viz_batch_indices(batch_size):
                env_b = {
                    'env':          inputs['env'][b],
                    'res':          res[b],
                    'extent':       inputs['extent'][b],
                    'origin_point': inputs['origin_point'][b],
                }

                self.plot_environment_rviz(env_b)
                self.debug_viz_state_action(object_aug_update, b, 'aug', color='white')
                stepper.step()
        return object_aug_update, local_origin_point_aug, local_center_aug

    def compute_collision_free_point_ik(self,
                                        default_robot_state,
                                        points,
                                        group_name,
                                        tip_names,
                                        scene_msg,
                                        ik_params):
        pass

    def aug_ik(self,
               inputs_aug: Dict,
               default_robot_positions,
               ik_params: IkParams,
               batch_size: int):
        """

        Args:
            inputs_aug: a dict containing the desired gripper positions as well as the scene_msg and other state info
            default_robot_positions: default robot joint state to seed IK
            batch_size:

        Returns:

        """
        pass

    def plot_aug_points_rviz(self, obj_i: int, obj_points_b_i, label: str, color_map):
        obj_points_b_i_time = tf.reshape(obj_points_b_i, [-1, 8, 3])
        blocks_aug_msg = MarkerArray()
        for t, obj_points_b_i_t in enumerate(obj_points_b_i_time):
            color_t = ColorRGBA(*color_map(t / obj_points_b_i_time.shape[0]))
            color_t.a = 0.2

            block_center = tf.reduce_mean(obj_points_b_i_t, 0)
            block_x_axis = obj_points_b_i_t[3] - obj_points_b_i_t[0]  # [3]
            block_y_axis = obj_points_b_i_t[1] - obj_points_b_i_t[0]  # [3]
            block_z_axis = obj_points_b_i_t[4] - obj_points_b_i_t[0]  # [3]
            block_x_axis_norm, _ = tf.linalg.normalize(block_x_axis)
            block_y_axis_norm, _ = tf.linalg.normalize(block_y_axis)
            block_z_axis_norm, _ = tf.linalg.normalize(block_z_axis)
            block_rot_mat = tf.stack([block_x_axis_norm, block_y_axis_norm, block_z_axis_norm], axis=-1)
            block_quat = quaternion.from_rotation_matrix(block_rot_mat)

            block_aug_msg = Marker()
            block_aug_msg.ns = 'blocks_' + label
            block_aug_msg.id = t + 1000 * obj_i
            block_aug_msg.header.frame_id = 'world'
            block_aug_msg.action = Marker.ADD
            block_aug_msg.type = Marker.CUBE
            block_aug_msg.scale.x = tf.linalg.norm(block_x_axis)
            block_aug_msg.scale.y = tf.linalg.norm(block_y_axis)
            block_aug_msg.scale.z = tf.linalg.norm(block_z_axis)
            block_aug_msg.pose.position.x = block_center[0]
            block_aug_msg.pose.position.y = block_center[1]
            block_aug_msg.pose.position.z = block_center[2]
            block_aug_msg.pose.orientation.x = block_quat[0]
            block_aug_msg.pose.orientation.y = block_quat[1]
            block_aug_msg.pose.orientation.z = block_quat[2]
            block_aug_msg.pose.orientation.w = block_quat[3]
            block_aug_msg.color = color_t

            blocks_aug_msg.markers.append(block_aug_msg)
        self.blocks_aug_pub.publish(blocks_aug_msg)

    def reset_planning_viz(self):
        super().reset_planning_viz()
        m = make_delete_markerarray(ns='blocks_aug')
        self.blocks_aug_pub.publish(m)

    def __repr__(self):
        return "blocks"
