from typing import Dict, Optional

import numpy as np
import tensorflow as tf
from matplotlib import colors

import ros_numpy
import rospy
from arc_utilities.listener import Listener
from arc_utilities.marker_utils import scale_marker_array
from geometry_msgs.msg import Point
from link_bot_data.dataset_utils import get_maybe_predicted, in_maybe_predicted, add_predicted
from link_bot_data.visualization import rviz_arrow
from link_bot_gazebo_python.gazebo_services import gz_scope
from link_bot_gazebo_python.gazebo_utils import get_gazebo_kinect_pose
from link_bot_gazebo_python.position_3d import Position3D
from link_bot_pycommon import grid_utils
from link_bot_pycommon.base_3d_scenario import Base3DScenario
from link_bot_pycommon.bbox_marker_utils import make_box_marker_from_extents
from link_bot_pycommon.bbox_visualization import extent_array_to_bbox
from link_bot_pycommon.collision_checking import inflate_tf_3d
from link_bot_pycommon.constants import KINECT_MAX_DEPTH
from link_bot_pycommon.make_rope_markers import make_gripper_marker, make_rope_marker
from link_bot_pycommon.marker_index_generator import marker_index_generator
from link_bot_pycommon.pycommon import default_if_none
from link_bot_pycommon.ros_pycommon import publish_color_image, publish_depth_image, get_camera_params, \
    transform_points_to_robot_frame
from moonshine.base_learned_dynamics_model import dynamics_loss_function, dynamics_points_metrics_function
from moonshine.moonshine_utils import numpify
from peter_msgs.srv import *
from rosgraph.names import ns_join
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from std_msgs.msg import Float32
from std_srvs.srv import Empty, EmptyRequest
from tf import transformations
from visualization_msgs.msg import MarkerArray, Marker

try:
    from jsk_recognition_msgs.msg import BoundingBox
except ImportError:
    rospy.logwarn("ignoring failed import of BBox message")

rope_key_name = 'rope'


class FloatingRopeScenario(Base3DScenario):
    IMAGE_H = 90
    IMAGE_W = 160
    n_links = 25
    KINECT_NAME = "kinect2"
    KINECT_CHANNEL = "qhd"
    COLOR_IMAGE_TOPIC = ns_join(KINECT_NAME, ns_join(KINECT_CHANNEL, "image_color_rect"))
    DEPTH_IMAGE_TOPIC = ns_join(KINECT_NAME, ns_join(KINECT_CHANNEL, "image_depth_rect"))
    CAMERA_INFO_TOPIC = ns_join(KINECT_NAME, ns_join(KINECT_CHANNEL, "camera_info"))
    crop_region = {
        'min_y': 0,
        'min_x': 0,
        'max_y': 540,
        'max_x': 960,
    }
    ROPE_NAMESPACE = 'rope_3d'

    # TODO: break out the different pieces of get_state to make them composable,
    #  since there are just a few shared amongst all the scenarios
    # TODO: about this... maybe they should all be pure functions? do we really need "self" at all?
    def __init__(self):
        super().__init__()
        self.color_image_listener = Listener(self.COLOR_IMAGE_TOPIC, Image)
        self.depth_image_listener = Listener(self.DEPTH_IMAGE_TOPIC, Image)
        self.camera_info_listener = Listener(self.CAMERA_INFO_TOPIC, CameraInfo)
        self.state_color_viz_pub = rospy.Publisher("state_color_viz", Image, queue_size=10, latch=True)
        self.state_depth_viz_pub = rospy.Publisher("state_depth_viz", Image, queue_size=10, latch=True)
        self.last_action = None
        self.get_rope_end_points_srv = rospy.ServiceProxy(ns_join(self.ROPE_NAMESPACE, "get_dual_gripper_points"),
                                                          GetDualGripperPoints)
        self.get_rope_srv = rospy.ServiceProxy(ns_join(self.ROPE_NAMESPACE, "get_rope_state"), GetRopeState)

        self.pos3d = Position3D()
        self.cdcpd_listener = Listener("cdcpd/output", PointCloud2)
        self.set_rope_state_srv = rospy.ServiceProxy(ns_join(self.ROPE_NAMESPACE, "set_rope_state"), SetRopeState)
        self.reset_srv = rospy.ServiceProxy("/gazebo/reset_simulation", Empty)

        try:
            self.left_gripper_bbox_pub = rospy.Publisher('/left_gripper_bbox_pub', BoundingBox, queue_size=10,
                                                         latch=True)
            self.right_gripper_bbox_pub = rospy.Publisher('/right_gripper_bbox_pub', BoundingBox, queue_size=10,
                                                          latch=True)
        except NameError:
            pass
        self.overstretching_srv = rospy.ServiceProxy(ns_join(self.ROPE_NAMESPACE, "rope_overstretched"),
                                                     GetOverstretching)
        self.error_pub = rospy.Publisher("error", Float32, queue_size=10)

        self.max_action_attempts = 100

        self.robot_reset_rng = np.random.RandomState(0)

    def needs_reset(self, state: Dict, params: Dict):
        return self.is_rope_overstretched()

    def is_rope_overstretched(self):
        res: GetOverstretchingResponse = self.overstretching_srv(GetOverstretchingRequest())
        return res.overstretched

    def trajopt_distance_to_goal_differentiable(self, final_state: Dict, goal_state: Dict):
        return self.cfm_distance(final_state['z'], goal_state['z'])

    def trajopt_distance_differentiable(self, s1, s2):
        return self.cfm_distance(s1['z'], s2['z'])

    def cfm_distance(self, z1, z2):
        return tf.math.reduce_sum(tf.math.square(z1 - z2), axis=-1, keepdims=True)

    def get_environment(self, params: Dict, **kwargs):
        return {}

    def hard_reset(self):
        self.reset_srv(EmptyRequest())

    def on_before_action(self):
        self.register_fake_grasping()

    def on_before_data_collection(self, params: Dict):
        self.on_before_action()

        left_gripper_position = np.array([1.0, 0.2, 1.0])
        right_gripper_position = np.array([1.0, -0.2, 1.0])
        init_action = {
            'left_gripper_position':  left_gripper_position,
            'right_gripper_position': right_gripper_position,
        }
        self.execute_action(init_action)

    def execute_action(self, action: Dict):
        speed_mps = action.get('speed', 0.1)
        left_req = Position3DActionRequest(speed_mps=speed_mps,
                                           scoped_link_name=gz_scope(self.ROPE_NAMESPACE, 'left_gripper'),
                                           position=ros_numpy.msgify(Point, action['left_gripper_position']))
        right_req = Position3DActionRequest(speed_mps=speed_mps,
                                            scoped_link_name=gz_scope(self.ROPE_NAMESPACE, 'right_gripper'),
                                            position=ros_numpy.msgify(Point, action['right_gripper_position']))
        self.pos3d.set(left_req)
        self.pos3d.set(right_req)

        wait_req = Position3DWaitRequest()
        wait_req.timeout_s = 10.0
        wait_req.scoped_link_names.append(gz_scope(self.ROPE_NAMESPACE, 'left_gripper'))
        wait_req.scoped_link_names.append(gz_scope(self.ROPE_NAMESPACE, 'right_gripper'))
        self.pos3d.wait(wait_req)

        rope_settling_time = action.get('settling_time', 1.0)
        rospy.sleep(rope_settling_time)

    def reset_rope(self, action_params: Dict):
        reset = SetRopeStateRequest()

        # TODO: rename this to rope endpoints reset positions or something
        reset.left_gripper.x = numpify(action_params['left_gripper_reset_position'][0])
        reset.left_gripper.y = numpify(action_params['left_gripper_reset_position'][1])
        reset.left_gripper.z = numpify(action_params['left_gripper_reset_position'][2])
        reset.right_gripper.x = numpify(action_params['right_gripper_reset_position'][0])
        reset.right_gripper.y = numpify(action_params['right_gripper_reset_position'][1])
        reset.right_gripper.z = numpify(action_params['right_gripper_reset_position'][2])

        self.set_rope_state_srv(reset)

    def dynamics_dataset_metadata(self):
        metadata = super().dynamics_dataset_metadata()
        kinect_pose = get_gazebo_kinect_pose()
        kinect_params = get_camera_params(self.KINECT_NAME)
        metadata.update({
            'world_to_rgb_optical_frame': self.tf.get_transform(parent='world', child='kinect2_rgb_optical_frame'),
            'kinect_pose':                ros_numpy.numpify(kinect_pose),
            'kinect_params':              kinect_params,
        })
        return metadata

    def sample_action(self,
                      action_rng: np.random.RandomState,
                      environment: Dict,
                      state: Dict,
                      action_params: Dict,
                      validate: bool,
                      stateless: Optional[bool] = False):
        self.viz_action_sample_bbox(self.left_gripper_bbox_pub, self.get_action_sample_extent(action_params, 'left'))
        self.viz_action_sample_bbox(self.right_gripper_bbox_pub, self.get_action_sample_extent(action_params, 'right'))

        action = None
        for _ in range(self.max_action_attempts):
            # move in the same direction as the previous action with some probability
            repeat_probability = action_params['repeat_delta_gripper_motion_probability']
            if state.get('is_overstretched', False):
                left_gripper_delta_position = np.zeros(3, dtype=np.float)
                right_gripper_delta_position = np.zeros(3, dtype=np.float)
            elif not stateless and self.last_action is not None and action_rng.uniform(0, 1) < repeat_probability:
                left_gripper_delta_position = self.last_action['left_gripper_delta_position']
                right_gripper_delta_position = self.last_action['right_gripper_delta_position']
            else:
                # Sample a new random action
                left_gripper_delta_position = self.sample_delta_position(action_params, action_rng)
                right_gripper_delta_position = self.sample_delta_position(action_params, action_rng)

            # Apply delta and check for out of bounds
            left_gripper_position = state['left_gripper'] + left_gripper_delta_position
            right_gripper_position = state['right_gripper'] + right_gripper_delta_position

            action = {
                'left_gripper_position':        left_gripper_position,
                'right_gripper_position':       right_gripper_position,
                'left_gripper_delta_position':  left_gripper_delta_position,
                'right_gripper_delta_position': right_gripper_delta_position,
            }

            if not validate or self.is_action_valid(action, action_params):
                self.last_action = action
                return action

        rospy.logwarn("Could not find a valid action, executing an invalid one")
        return action

    def is_action_valid(self, action: Dict, action_params: Dict):
        out_of_bounds = FloatingRopeScenario.grippers_out_of_bounds(action['left_gripper_position'],
                                                                    action['right_gripper_position'],
                                                                    action_params)

        max_gripper_d = default_if_none(action_params['max_distance_between_grippers'], 1000)
        gripper_d = np.linalg.norm(action['left_gripper_position'] - action['right_gripper_position'])
        too_far = gripper_d > max_gripper_d

        return not out_of_bounds and not too_far

    def get_action_sample_extent(self, action_params: Dict, prefix: str):
        k = prefix + '_gripper_action_sample_extent'
        if k in action_params:
            gripper_extent = np.array(action_params[k]).reshape([3, 2])
        else:
            gripper_extent = np.array(action_params['extent']).reshape([3, 2])
        return gripper_extent

    def viz_action_sample_bbox(self, gripper_bbox_pub: rospy.Publisher, gripper_extent):
        gripper_bbox_msg = extent_array_to_bbox(gripper_extent)
        gripper_bbox_msg.header.frame_id = 'world'
        gripper_bbox_pub.publish(gripper_bbox_msg)

    def sample_delta_position(self, action_params, action_rng):
        pitch = action_rng.uniform(-np.pi, np.pi)
        yaw = action_rng.uniform(-np.pi, np.pi)
        displacement = action_rng.uniform(0, action_params['max_distance_gripper_can_move'])
        rotation_matrix = transformations.euler_matrix(0, pitch, yaw)
        gripper_delta_position_homo = rotation_matrix @ np.array([1, 0, 0, 1]) * displacement
        gripper_delta_position = gripper_delta_position_homo[:3]
        return gripper_delta_position

    @staticmethod
    def grippers_out_of_bounds(left_gripper, right_gripper, action_params: Dict):
        left_gripper_extent = action_params['left_gripper_action_sample_extent']
        right_gripper_extent = action_params['right_gripper_action_sample_extent']
        return FloatingRopeScenario.is_out_of_bounds(left_gripper, left_gripper_extent) \
               or FloatingRopeScenario.is_out_of_bounds(right_gripper, right_gripper_extent)

    @staticmethod
    def is_out_of_bounds(p, extent):
        x, y, z = p
        x_min, x_max, y_min, y_max, z_min, z_max = extent
        return x < x_min or x > x_max \
               or y < y_min or y > y_max \
               or z < z_min or z > z_max

    @staticmethod
    def interpolate(start_state, end_state, step_size=0.05):
        left_gripper_start = np.array(start_state['left_gripper'])
        left_gripper_end = np.array(end_state['left_gripper'])

        right_gripper_start = np.array(start_state['right_gripper'])
        right_gripper_end = np.array(end_state['right_gripper'])

        left_gripper_delta = left_gripper_end - left_gripper_start
        right_gripper_delta = right_gripper_end - right_gripper_start

        left_gripper_steps = np.round(np.linalg.norm(left_gripper_delta) / step_size).astype(np.int64)
        right_gripper_steps = np.round(np.linalg.norm(right_gripper_delta) / step_size).astype(np.int64)
        steps = max(left_gripper_steps, right_gripper_steps)

        interpolated_actions = []
        for t in np.linspace(step_size, 1, steps):
            left_gripper_i = left_gripper_start + left_gripper_delta * t
            right_gripper_i = right_gripper_start + right_gripper_delta * t
            action = {
                'left_gripper_position':  left_gripper_i,
                'right_gripper_position': right_gripper_i,
            }
            interpolated_actions.append(action)

        return interpolated_actions

    @staticmethod
    def robot_name():
        return "rope_3d"

    def randomize_environment(self, env_rng, params: Dict):
        pass

    @staticmethod
    def put_state_robot_frame(state: Dict):
        rope = state[rope_key_name]
        rope_points_shape = rope.shape[:-1].as_list() + [-1, 3]
        rope_points = tf.reshape(rope, rope_points_shape)

        # This assumes robot is at 0 0 0
        robot_position = tf.constant([[0, 0, 0]], tf.float32)
        left_gripper_robot = state['left_gripper']
        right_gripper_robot = state['right_gripper']

        rope_points_robot = rope_points - tf.expand_dims(robot_position, axis=-2)
        rope_robot = tf.reshape(rope_points_robot, rope.shape)

        return {
            'left_gripper':  left_gripper_robot,
            'right_gripper': right_gripper_robot,
            rope_key_name:   rope_robot,
        }

    @staticmethod
    def put_state_local_frame(state: Dict):
        rope = state[rope_key_name]
        rope_points_shape = rope.shape[:-1].as_list() + [-1, 3]
        rope_points = tf.reshape(rope, rope_points_shape)

        center = tf.reduce_mean(rope_points, axis=-2)

        left_gripper_local = state['left_gripper'] - center
        right_gripper_local = state['right_gripper'] - center

        rope_points_local = rope_points - tf.expand_dims(center, axis=-2)
        rope_local = tf.reshape(rope_points_local, rope.shape)

        return {
            'left_gripper':  left_gripper_local,
            'right_gripper': right_gripper_local,
            rope_key_name:   rope_local,
        }

    @staticmethod
    def local_environment_center_differentiable(state):
        rope_vector = state[rope_key_name]
        rope_points = tf.reshape(rope_vector, [rope_vector.shape[0], -1, 3])
        center = tf.reduce_mean(rope_points, axis=1)
        return center

    @staticmethod
    def apply_local_action_at_state(state, local_action):
        return {
            'left_gripper_position':  state['left_gripper'] + local_action['left_gripper_delta'],
            'right_gripper_position': state['right_gripper'] + local_action['right_gripper_delta']
        }

    @staticmethod
    def add_noise(action: Dict, noise_rng: np.random.RandomState):
        left_gripper_noise = noise_rng.normal(scale=0.01, size=[3])
        right_gripper_noise = noise_rng.normal(scale=0.01, size=[3])
        return {
            'left_gripper_position':  action['left_gripper_position'] + left_gripper_noise,
            'right_gripper_position': action['right_gripper_position'] + right_gripper_noise
        }

    @staticmethod
    def integrate_dynamics(s_t: Dict, delta_s_t: Dict):
        return {k: s_t[k] + delta_s_t[k] for k in s_t.keys()}

    @staticmethod
    def put_action_local_frame(state: Dict, action: Dict):
        target_left_gripper_position = action['left_gripper_position']
        target_right_gripper_position = action['right_gripper_position']
        n_action = target_left_gripper_position.shape[1]

        current_left_gripper_point = state['left_gripper']
        current_right_gripper_point = state['right_gripper']

        left_gripper_delta = target_left_gripper_position - current_left_gripper_point[:, :n_action]
        right_gripper_delta = target_right_gripper_position - current_right_gripper_point[:, :n_action]

        return {
            'left_gripper_delta':  left_gripper_delta,
            'right_gripper_delta': right_gripper_delta,
        }

    @staticmethod
    def state_to_gripper_position(state: Dict):
        gripper_position1 = np.reshape(state['left_gripper'], [3])
        gripper_position2 = np.reshape(state['right_gripper'], [3])
        return gripper_position1, gripper_position2

    def get_cdcpd_state(self):
        cdcpd_msg: PointCloud2 = self.cdcpd_listener.get()

        points = transform_points_to_robot_frame(self.tf, cdcpd_msg)

        cdcpd_vector = points.flatten()
        return cdcpd_vector

    def get_rope_state(self):
        rope_res = self.get_rope_srv(GetRopeStateRequest())
        rope_state_vector = []
        for p in rope_res.positions:
            rope_state_vector.append(p.x)
            rope_state_vector.append(p.y)
            rope_state_vector.append(p.z)
        rope_velocity_vector = []
        for v in rope_res.velocities:
            rope_velocity_vector.append(v.x)
            rope_velocity_vector.append(v.y)
            rope_velocity_vector.append(v.z)
        return rope_state_vector

    def is_rope_point_attached(self, gripper: str):
        scoped_link_name = gz_scope(self.ROPE_NAMESPACE, gripper + '_gripper')
        res: GetPosition3DResponse = self.pos3d.get(scoped_link_name=scoped_link_name)
        return res.enabled

    def get_rope_point_position(self, gripper: str):
        # NOTE: consider getting rid of this message type/service just use rope state [0] and rope state [-1]
        #  although that looses semantic meaning and means hard-coding indices a lot...
        scoped_link_name = gz_scope(self.ROPE_NAMESPACE, gripper + '_gripper')
        res: GetPosition3DResponse = self.pos3d.get(scoped_link_name=scoped_link_name)
        rope_point_position = ros_numpy.numpify(res.pos)
        return rope_point_position

    def get_rope_point_positions(self):
        return self.get_rope_point_position('left'), self.get_rope_point_position('right')

    def get_state(self):
        color_depth_cropped = self.get_rgbd()

        rope_state_vector = self.get_rope_state()
        # cdcpd_vector = self.get_cdcpd_state()
        left_rope_point_position, right_rope_point_position = self.get_rope_point_positions()

        return {
            'left_gripper':     left_rope_point_position,
            'right_gripper':    right_rope_point_position,
            'gt_rope':          np.array(rope_state_vector, np.float32),
            # rope_key_name:   np.array(cdcpd_vector, np.float32),
            'rgbd':             color_depth_cropped,
            'is_overstretched': self.is_rope_overstretched(),
        }

    def get_rgbd(self):
        color_msg: Image = self.color_image_listener.get()
        depth_msg = self.depth_image_listener.get()

        depth = np.expand_dims(ros_numpy.numpify(depth_msg), axis=-1)
        bgr = ros_numpy.numpify(color_msg)
        rgb = np.flip(bgr, axis=2)

        # NaN Depths means out of range, so clip to the max range
        depth = np.clip(np.nan_to_num(depth, nan=KINECT_MAX_DEPTH), 0, KINECT_MAX_DEPTH)
        rgbd = np.concatenate([rgb, depth], axis=2)

        # box = tf.convert_to_tensor([self.crop_region['min_y'] / rgb.shape[0],
        #                             self.crop_region['min_x'] / rgb.shape[1],
        #                             self.crop_region['max_y'] / rgb.shape[0],
        #                             self.crop_region['max_x'] / rgb.shape[1]], dtype=tf.float32)
        # this operates on a batch
        # rgbd_cropped = tf.image.crop_and_resize(image=tf.expand_dims(rgbd, axis=0),
        #                                         boxes=tf.expand_dims(box, axis=0),
        #                                         box_indices=[0],
        #                                         crop_size=[self.IMAGE_H, self.IMAGE_W])
        # rgbd_cropped = remove_batch(rgbd_cropped)

        def _debug_show_image(_rgb_depth_cropped):
            import matplotlib.pyplot as plt
            plt.imshow(tf.cast(_rgb_depth_cropped[:, :, :3], tf.int32))
            plt.show()

        # BEGIN DEBUG
        # _debug_show_image(rgbd_cropped)
        # END DEBUG
        # return rgbd_cropped.numpy()
        return rgbd

    def observations_description(self) -> Dict:
        return {
            'left_gripper':  3,
            'right_gripper': 3,
            'rgbd':          [self.IMAGE_H, self.IMAGE_W, 4],
        }

    @staticmethod
    def states_description() -> Dict:
        return {
        }

    @staticmethod
    def observation_features_description() -> Dict:
        return {
            rope_key_name: FloatingRopeScenario.n_links * 3,
            'cdcpd':       FloatingRopeScenario.n_links * 3,
        }

    @staticmethod
    def actions_description() -> Dict:
        # should match the keys of the dict return from action_to_dataset_action
        return {
            'left_gripper_position':  3,
            'right_gripper_position': 3,
        }

    @staticmethod
    def state_to_points_for_cc(state: Dict):
        return state[rope_key_name].reshape(-1, 3)

    def __repr__(self):
        return "DualFloatingGripperRope"

    @staticmethod
    def simple_name():
        return "dual_floating"

    @staticmethod
    def sample_gripper_goal(environment: Dict, rng: np.random.RandomState, planner_params: Dict):
        env_inflated = inflate_tf_3d(env=environment['env'],
                                     radius_m=planner_params['goal_params']['threshold'], res=environment['res'])
        goal_extent = planner_params['goal_params']['extent']

        while True:
            extent = np.array(goal_extent).reshape(3, 2)
            left_gripper = rng.uniform(extent[:, 0], extent[:, 1])
            right_gripper = rng.uniform(extent[:, 0], extent[:, 1])
            goal = {
                'left_gripper':  left_gripper,
                'right_gripper': right_gripper,
            }
            row1, col1, channel1 = grid_utils.point_to_idx_3d_in_env(
                left_gripper[0], left_gripper[1], left_gripper[2], environment)
            row2, col2, channel2 = grid_utils.point_to_idx_3d_in_env(
                right_gripper[0], right_gripper[1], right_gripper[2], environment)
            collision1 = env_inflated[row1, col1, channel1] > 0.5
            collision2 = env_inflated[row2, col2, channel2] > 0.5
            if not collision1 and not collision2:
                return goal

    def sample_goal(self, environment: Dict, rng: np.random.RandomState, planner_params: Dict):
        goal_type = planner_params['goal_params']['goal_type']
        if goal_type == 'midpoint':
            return self.sample_midpoint_goal(environment, rng, planner_params)
        else:
            raise NotImplementedError(planner_params['goal_params']['goal_type'])

    def sample_midpoint_goal(self, environment: Dict, rng: np.random.RandomState, planner_params: Dict):
        goal_extent = planner_params['goal_params']['extent']

        if environment == {}:
            rospy.loginfo("Assuming no obstacles in the environment")
            extent = np.array(goal_extent).reshape(3, 2)
            p = rng.uniform(extent[:, 0], extent[:, 1])
            goal = {'midpoint': p}
            return goal

        env_inflated = inflate_tf_3d(env=environment['env'],
                                     radius_m=planner_params['goal_params']['threshold'], res=environment['res'])
        # DEBUG visualize the inflated env
        # from copy import deepcopy
        # environment_ = deepcopy(environment)
        # environment_['env'] = env_inflated
        # self.plot_environment_rviz(environment_)
        # END DEBUG

        while True:
            extent = np.array(goal_extent).reshape(3, 2)
            p = rng.uniform(extent[:, 0], extent[:, 1])
            goal = {'midpoint': p}
            row, col, channel = grid_utils.point_to_idx_3d_in_env(p[0], p[1], p[2], environment)
            collision = env_inflated[row, col, channel] > 0.5
            if not collision:
                return goal

    @staticmethod
    def distance_to_gripper_goal(state: Dict, goal: Dict):
        left_gripper = state['left_gripper']
        right_gripper = state['right_gripper']
        distance1 = tf.linalg.norm(goal['left_gripper'] - left_gripper)
        distance2 = tf.linalg.norm(goal['right_gripper'] - right_gripper)
        return tf.math.maximum(distance1, distance2)

    @staticmethod
    def distance_grippers_and_any_point_goal(state: Dict, goal: Dict):
        rope_points = tf.reshape(state[rope_key_name], [-1, 3])
        # well ok not _any_ node, but ones near the middle
        n_from_ends = 5
        distances = tf.linalg.norm(tf.expand_dims(goal['point'], axis=0) -
                                   rope_points, axis=1)[n_from_ends:-n_from_ends]
        rope_distance = tf.min(distances)

        left_gripper = state['left_gripper']
        right_gripper = state['right_gripper']
        distance1 = tf.linalg.norm(goal['left_gripper'] - left_gripper)
        distance2 = tf.linalg.norm(goal['right_gripper'] - right_gripper)
        return tf.math.maximum(tf.math.maximum(distance1, distance2), rope_distance)

    @staticmethod
    def distance_to_any_point_goal(state: Dict, goal: Dict):
        rope_points = tf.reshape(state[rope_key_name], [-1, 3])
        # well ok not _any_ node, but ones near the middle
        n_from_ends = 7
        distances = tf.linalg.norm(tf.expand_dims(goal['point'], axis=0) -
                                   rope_points, axis=1)[n_from_ends:-n_from_ends]
        min_distance = tf.min(distances)
        return min_distance

    @staticmethod
    def distance_to_midpoint_goal(state: Dict, goal: Dict):
        rope_points = tf.reshape(state[rope_key_name], [-1, 3])
        rope_midpoint = rope_points[int(FloatingRopeScenario.n_links / 2)]
        distance = tf.linalg.norm(goal['midpoint'] - rope_midpoint)
        return distance

    def distance_to_goal(self, state: Dict, goal: Dict):
        if goal['goal_type'] == 'midpoint':
            return self.distance_to_midpoint_goal(state, goal)
        elif goal['goal_type'] == 'any_point':
            return self.distance_to_any_point_goal(state, goal)
        elif goal['goal_type'] == 'grippers':
            return self.distance_to_gripper_goal(state, goal)
        elif goal['goal_type'] == 'grippers_and_point':
            return self.distance_grippers_and_any_point_goal(state, goal)
        else:
            raise NotImplementedError()

    def goal_state_to_goal(self, goal_state: Dict, goal_type: str):
        if goal_type == 'midpoint':
            rope_points = tf.reshape(goal_state[rope_key_name], [-1, 3])
            rope_midpoint = rope_points[int(FloatingRopeScenario.n_links / 2)]
            return {
                'goal_type': goal_type,
                'midpoint':  rope_midpoint,
            }
        elif goal_type == 'any_point':
            # NOTE: since all points on the sampled ropes are the same point, it doesn't matter which one we pick here
            rope_points = tf.reshape(goal_state[rope_key_name], [-1, 3])
            rope_point = rope_points[0]
            return {
                'goal_type': goal_type,
                'point':     rope_point,
            }
        elif goal_type == 'grippers':
            left_gripper = goal_state['left_gripper']
            right_gripper = goal_state['right_gripper']
            return {
                'goal_type':     goal_type,
                'left_gripper':  left_gripper,
                'right_gripper': right_gripper,
            }
        elif goal_type == 'grippers_and_point':
            left_gripper = goal_state['left_gripper']
            right_gripper = goal_state['right_gripper']
            rope_points = tf.reshape(goal_state[rope_key_name], [-1, 3])
            rope_point = rope_points[0]
            return {
                'goal_type':     goal_type,
                'point':         rope_point,
                'left_gripper':  left_gripper,
                'right_gripper': right_gripper,
            }
        else:
            raise NotImplementedError()

    def classifier_distance(self, s1: Dict, s2: Dict):
        model_error = np.linalg.norm(s1[rope_key_name] - s2[rope_key_name], axis=-1)
        # labeling_states = s1['rope']
        # labeling_predicted_states = s2['rope']
        # points_shape = labeling_states.shape.as_list()[:2] + [-1, 3]
        # labeling_points = tf.reshape(labeling_states, points_shape)
        # labeling_predicted_points = tf.reshape(labeling_predicted_states, points_shape)
        # model_error = tf.reduce_mean(tf.linalg.norm(labeling_points - labeling_predicted_points, axis=-1), axis=-1)
        return model_error

    def compute_label(self, actual: Dict, predicted: Dict, labeling_params: Dict):
        # NOTE: this should be using the same distance metric as the planning, which should also be the same as the labeling
        # done when making the classifier dataset
        actual_rope = np.array(actual["rope"])
        predicted_rope = np.array(predicted["rope"])
        model_error = np.linalg.norm(actual_rope - predicted_rope)
        threshold = labeling_params['threshold']
        is_close = model_error < threshold
        return is_close

    def plot_goal_rviz(self, goal: Dict, goal_threshold: float, actually_at_goal: Optional[bool] = None):
        if actually_at_goal:
            r = 0.4
            g = 0.8
            b = 0.4
            a = 0.6
        else:
            r = 0.5
            g = 0.3
            b = 0.8
            a = 0.6

        goal_marker_msg = MarkerArray()

        if 'midpoint' in goal:
            midpoint_marker = Marker()
            midpoint_marker.scale.x = goal_threshold * 2
            midpoint_marker.scale.y = goal_threshold * 2
            midpoint_marker.scale.z = goal_threshold * 2
            midpoint_marker.action = Marker.ADD
            midpoint_marker.type = Marker.SPHERE
            midpoint_marker.header.frame_id = "world"
            midpoint_marker.header.stamp = rospy.Time.now()
            midpoint_marker.ns = 'goal'
            midpoint_marker.id = 0
            midpoint_marker.color.r = r
            midpoint_marker.color.g = g
            midpoint_marker.color.b = b
            midpoint_marker.color.a = a
            midpoint_marker.pose.position.x = goal['midpoint'][0]
            midpoint_marker.pose.position.y = goal['midpoint'][1]
            midpoint_marker.pose.position.z = goal['midpoint'][2]
            midpoint_marker.pose.orientation.w = 1
            goal_marker_msg.markers.append(midpoint_marker)

        if 'point' in goal:
            point_marker = Marker()
            point_marker.scale.x = goal_threshold * 2
            point_marker.scale.y = goal_threshold * 2
            point_marker.scale.z = goal_threshold * 2
            point_marker.action = Marker.ADD
            point_marker.type = Marker.SPHERE
            point_marker.header.frame_id = "world"
            point_marker.header.stamp = rospy.Time.now()
            point_marker.ns = 'goal'
            point_marker.id = 0
            point_marker.color.r = r
            point_marker.color.g = g
            point_marker.color.b = b
            point_marker.color.a = a
            point_marker.pose.position.x = goal['point'][0]
            point_marker.pose.position.y = goal['point'][1]
            point_marker.pose.position.z = goal['point'][2]
            point_marker.pose.orientation.w = 1
            goal_marker_msg.markers.append(point_marker)

        if 'left_gripper' in goal:
            left_gripper_marker = Marker()
            left_gripper_marker.scale.x = goal_threshold * 2
            left_gripper_marker.scale.y = goal_threshold * 2
            left_gripper_marker.scale.z = goal_threshold * 2
            left_gripper_marker.action = Marker.ADD
            left_gripper_marker.type = Marker.SPHERE
            left_gripper_marker.header.frame_id = "world"
            left_gripper_marker.header.stamp = rospy.Time.now()
            left_gripper_marker.ns = 'goal'
            left_gripper_marker.id = 1
            left_gripper_marker.color.r = r
            left_gripper_marker.color.g = g
            left_gripper_marker.color.b = b
            left_gripper_marker.color.a = a
            left_gripper_marker.pose.position.x = goal['left_gripper'][0]
            left_gripper_marker.pose.position.y = goal['left_gripper'][1]
            left_gripper_marker.pose.position.z = goal['left_gripper'][2]
            left_gripper_marker.pose.orientation.w = 1
            goal_marker_msg.markers.append(left_gripper_marker)

        if 'right_gripper' in goal:
            right_gripper_marker = Marker()
            right_gripper_marker.scale.x = goal_threshold * 2
            right_gripper_marker.scale.y = goal_threshold * 2
            right_gripper_marker.scale.z = goal_threshold * 2
            right_gripper_marker.action = Marker.ADD
            right_gripper_marker.type = Marker.SPHERE
            right_gripper_marker.header.frame_id = "world"
            right_gripper_marker.header.stamp = rospy.Time.now()
            right_gripper_marker.ns = 'goal'
            right_gripper_marker.id = 2
            right_gripper_marker.color.r = r
            right_gripper_marker.color.g = g
            right_gripper_marker.color.b = b
            right_gripper_marker.color.a = a
            right_gripper_marker.pose.position.x = goal['right_gripper'][0]
            right_gripper_marker.pose.position.y = goal['right_gripper'][1]
            right_gripper_marker.pose.position.z = goal['right_gripper'][2]
            right_gripper_marker.pose.orientation.w = 1
            goal_marker_msg.markers.append(right_gripper_marker)

        self.state_viz_pub.publish(goal_marker_msg)

    def plot_goal_boxes(self, goal: Dict, goal_threshold: float, actually_at_goal: Optional[bool] = None):
        if actually_at_goal:
            r = 0.4
            g = 0.8
            b = 0.4
            a = 0.6
        else:
            r = 0.5
            g = 0.3
            b = 0.8
            a = 0.6

        goal_marker_msg = MarkerArray()

        if 'point_box' in goal:
            point_marker = make_box_marker_from_extents(goal['point_box'])
            point_marker.header.frame_id = "world"
            point_marker.header.stamp = rospy.Time.now()
            point_marker.ns = 'goal'
            point_marker.id = 0
            point_marker.color.r = r
            point_marker.color.g = g
            point_marker.color.b = b
            point_marker.color.a = a
            goal_marker_msg.markers.append(point_marker)

        if 'left_gripper_box' in goal:
            left_gripper_marker = make_box_marker_from_extents(goal['left_gripper_box'])
            left_gripper_marker.header.frame_id = "world"
            left_gripper_marker.header.stamp = rospy.Time.now()
            left_gripper_marker.ns = 'goal'
            left_gripper_marker.id = 1
            left_gripper_marker.color.r = r
            left_gripper_marker.color.g = g
            left_gripper_marker.color.b = b
            left_gripper_marker.color.a = a
            goal_marker_msg.markers.append(left_gripper_marker)

        if 'right_gripper_box' in goal:
            right_gripper_marker = make_box_marker_from_extents(goal['right_gripper_box'])
            right_gripper_marker.header.frame_id = "world"
            right_gripper_marker.header.stamp = rospy.Time.now()
            right_gripper_marker.ns = 'goal'
            right_gripper_marker.id = 2
            right_gripper_marker.color.r = r
            right_gripper_marker.color.g = g
            right_gripper_marker.color.b = b
            right_gripper_marker.color.a = a
            goal_marker_msg.markers.append(right_gripper_marker)

        self.state_viz_pub.publish(goal_marker_msg)

    @staticmethod
    def dynamics_loss_function(dataset_element, predictions):
        return dynamics_loss_function(dataset_element, predictions)

    @staticmethod
    def dynamics_metrics_function(dataset_element, predictions):
        return dynamics_points_metrics_function(dataset_element, predictions)

    def plot_tree_action(self, state: Dict, action: Dict, **kwargs):
        r = kwargs.pop("r", 0.2)
        g = kwargs.pop("g", 0.2)
        b = kwargs.pop("b", 0.8)
        a = kwargs.pop("a", 1.0)
        ig = marker_index_generator(self.tree_action_idx)
        idx1 = next(ig)
        idx2 = next(ig)
        self.plot_action_rviz(state, action, label='tree', color=[r, g, b, a], idx1=idx1, idx2=idx2, **kwargs)
        self.tree_action_idx += 1

    def plot_state_rviz(self, state: Dict, label: str, **kwargs):
        r, g, b, a = colors.to_rgba(kwargs.get("color", "r"))
        idx = kwargs.get("idx", 0)

        msg = MarkerArray()

        ig = marker_index_generator(idx)

        if 'gt_rope' in state:
            rope_points = np.reshape(state['gt_rope'], [-1, 3])
            markers = make_rope_marker(rope_points, 'world', label + "_gt_rope", next(ig), r, g, b, a, 0.01)
            msg.markers.extend(markers)

        if 'rope' in state:
            rope_points = np.reshape(state['rope'], [-1, 3])
            markers = make_rope_marker(rope_points, 'world', label + "_rope", next(ig), r, g, b, a, 0.01)
            msg.markers.extend(markers)

        if add_predicted('rope') in state:
            rope_points = np.reshape(state[add_predicted('rope')], [-1, 3])
            markers = make_rope_marker(rope_points, 'world', label + "_pred_rope", next(ig), r, g, b, a, 0.01,
                                       Marker.CUBE_LIST)
            msg.markers.extend(markers)

        if 'left_gripper' in state:
            r = 0.2
            g = 0.2
            b = 0.8
            left_gripper_sphere = make_gripper_marker(state['left_gripper'], next(ig), r, g, b, a, label + '_l',
                                                      Marker.SPHERE)
            msg.markers.append(left_gripper_sphere)

        if 'right_gripper' in state:
            r = 0.8
            g = 0.8
            b = 0.2
            right_gripper_sphere = make_gripper_marker(state['right_gripper'], next(ig), r, g, b, a, label + "_r",
                                                       Marker.SPHERE)
            msg.markers.append(right_gripper_sphere)

        if add_predicted('left_gripper') in state:
            r = 0.2
            g = 0.2
            b = 0.8
            lgpp = state[add_predicted('left_gripper')]
            left_gripper_sphere = make_gripper_marker(lgpp, next(ig), r, g, b, a, label + "_lp", Marker.CUBE)
            msg.markers.append(left_gripper_sphere)

        if add_predicted('right_gripper') in state:
            r = 0.8
            g = 0.8
            b = 0.2
            rgpp = state[add_predicted('right_gripper')]
            right_gripper_sphere = make_gripper_marker(rgpp, next(ig), r, g, b, a, label + "_rp",
                                                       Marker.CUBE)
            msg.markers.append(right_gripper_sphere)

        s = kwargs.get("scale", 1.0)
        msg = scale_marker_array(msg, s)

        self.state_viz_pub.publish(msg)

        if in_maybe_predicted('rgbd', state):
            publish_color_image(self.state_color_viz_pub, state['rgbd'][:, :, :3])
            publish_depth_image(self.state_depth_viz_pub, state['rgbd'][:, :, 3])

        if add_predicted('stdev') in state:
            stdev_t = state[add_predicted('stdev')][0]
            self.plot_stdev(stdev_t)

        if 'error' in state:
            error_msg = Float32()
            error_t = state['error']
            error_msg.data = error_t
            self.error_pub.publish(error_msg)

    def plot_action_rviz(self, state: Dict, action: Dict, label: str = 'action', **kwargs):
        state_action = {}
        state_action.update(state)
        state_action.update(action)
        self.plot_action_rviz_internal(state_action, label=label, **kwargs)

    def plot_action_rviz_internal(self, data: Dict, label: str, **kwargs):
        r, g, b, a = colors.to_rgba(kwargs.get("color", "b"))
        s1 = np.reshape(get_maybe_predicted(data, 'left_gripper'), [3])
        s2 = np.reshape(get_maybe_predicted(data, 'right_gripper'), [3])
        a1 = np.reshape(get_maybe_predicted(data, 'left_gripper_position'), [3])
        a2 = np.reshape(get_maybe_predicted(data, 'right_gripper_position'), [3])

        idx = kwargs.pop("idx", None)
        ig = marker_index_generator(idx)
        if idx is not None:
            idx1 = next(ig)
            idx2 = next(ig)
        else:
            idx1 = kwargs.pop("idx1", 0)
            idx2 = kwargs.pop("idx2", 1)

        msg = MarkerArray()
        msg.markers.append(rviz_arrow(s1, a1, r, g, b, a, idx=idx1, label=label, **kwargs))
        msg.markers.append(rviz_arrow(s2, a2, r, g, b, a, idx=idx2, label=label, **kwargs))

        self.action_viz_pub.publish(msg)

    def register_fake_grasping(self):
        register_left_req = RegisterPosition3DControllerRequest()
        register_left_req.scoped_link_name = gz_scope(self.ROPE_NAMESPACE, "left_gripper")
        register_left_req.controller_type = "kinematic"
        self.pos3d.register(register_left_req)
        register_right_req = RegisterPosition3DControllerRequest()
        register_right_req.scoped_link_name = gz_scope(self.ROPE_NAMESPACE, "right_gripper")
        register_right_req.controller_type = "kinematic"
        self.pos3d.register(register_right_req)

    def make_rope_endpoints_follow_gripper(self):
        left_follow_req = Position3DFollowRequest()
        left_follow_req.scoped_link_name = gz_scope(self.ROPE_NAMESPACE, "left_gripper")
        left_follow_req.frame_id = "left_tool"
        self.pos3d.follow(left_follow_req)

        right_follow_req = Position3DFollowRequest()
        right_follow_req.scoped_link_name = gz_scope(self.ROPE_NAMESPACE, "right_gripper")
        right_follow_req.frame_id = "right_tool"
        self.pos3d.follow(right_follow_req)
