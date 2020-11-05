import warnings
from typing import Dict, Optional

import numpy as np
import tensorflow as tf
from matplotlib import colors

import ros_numpy
from arc_utilities.marker_utils import scale_marker_array
from arm_robots_msgs.msg import Points
from link_bot_data.link_bot_dataset_utils import get_maybe_predicted, in_maybe_predicted, add_predicted
from link_bot_pycommon.ros_pycommon import KINECT_MAX_DEPTH, publish_color_image, publish_depth_image
from rosgraph.names import ns_join
from std_msgs.msg import Float32

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    import ompl.base as ob
    import ompl.control as oc

import rospy
import tf2_sensor_msgs
from arc_utilities.listener import Listener
from arm_robots_msgs.srv import GrippersTrajectory, GrippersTrajectoryRequest
from geometry_msgs.msg import Point
from jsk_recognition_msgs.msg import BoundingBox
from link_bot_data.visualization import rviz_arrow
from link_bot_pycommon import grid_utils
from link_bot_pycommon.base_3d_scenario import Base3DScenario
from link_bot_pycommon.collision_checking import inflate_tf_3d
from link_bot_pycommon.grid_utils import extent_to_env_size, extent_to_center, extent_to_bbox, extent_array_to_bbox
from link_bot_pycommon.pycommon import default_if_none
from moonshine.base_learned_dynamics_model import dynamics_loss_function, dynamics_points_metrics_function
from moonshine.moonshine_utils import numpify, remove_batch
from peter_msgs.srv import GetDualGripperPoints, SetRopeState, SetRopeStateRequest, GetRopeState, GetRopeStateRequest, \
    GetDualGripperPointsRequest, GetDualGripperPointsResponse, GetOverstretchingResponse, GetOverstretchingRequest, \
    GetOverstretching
from sensor_msgs.msg import Image, PointCloud2
from std_srvs.srv import Empty, EmptyRequest
from tf import transformations
from visualization_msgs.msg import MarkerArray, Marker


def make_gripper_marker(position, id, r, g, b, a, label, type):
    gripper_marker = Marker()
    gripper_marker.action = Marker.ADD  # create or modify
    gripper_marker.type = type
    gripper_marker.header.frame_id = "world"
    gripper_marker.header.stamp = rospy.Time.now()
    gripper_marker.ns = label
    gripper_marker.id = id
    gripper_marker.scale.x = 0.02
    gripper_marker.scale.y = 0.02
    gripper_marker.scale.z = 0.02
    gripper_marker.pose.position.x = position[0]
    gripper_marker.pose.position.y = position[1]
    gripper_marker.pose.position.z = position[2]
    gripper_marker.pose.orientation.w = 1
    gripper_marker.color.r = r
    gripper_marker.color.g = g
    gripper_marker.color.b = b
    gripper_marker.color.a = a
    return gripper_marker


def make_rope_marker(rope_points, frame_id, label, idx, r, g, b, a, points_marker_type=Marker.SPHERE_LIST):
    lines = Marker()
    lines.action = Marker.ADD  # create or modify
    lines.type = Marker.LINE_STRIP
    lines.header.frame_id = frame_id
    lines.header.stamp = rospy.Time.now()
    lines.ns = label
    lines.id = 6 * idx + 0
    lines.pose.position.x = 0
    lines.pose.position.y = 0
    lines.pose.position.z = 0
    lines.pose.orientation.x = 0
    lines.pose.orientation.y = 0
    lines.pose.orientation.z = 0
    lines.pose.orientation.w = 1
    lines.scale.x = 0.005
    lines.scale.y = 0.005
    lines.scale.z = 0.005
    lines.color.r = r
    lines.color.g = g
    lines.color.b = b
    lines.color.a = a
    points_marker = Marker()
    points_marker.action = Marker.ADD  # create or modify
    points_marker.type = points_marker_type
    points_marker.header.frame_id = frame_id
    points_marker.header.stamp = rospy.Time.now()
    points_marker.ns = label
    points_marker.id = 6 * idx + 1
    points_marker.scale.x = 0.01
    points_marker.scale.y = 0.01
    points_marker.scale.z = 0.01
    points_marker.pose.position.x = 0
    points_marker.pose.position.y = 0
    points_marker.pose.position.z = 0
    points_marker.pose.orientation.x = 0
    points_marker.pose.orientation.y = 0
    points_marker.pose.orientation.z = 0
    points_marker.pose.orientation.w = 1
    points_marker.color.r = r
    points_marker.color.g = g
    points_marker.color.b = b
    points_marker.color.a = a
    for i, (x, y, z) in enumerate(rope_points):
        point = Point()
        point.x = x
        point.y = y
        point.z = z

        points_marker.points.append(point)
        lines.points.append(point)
    midpoint_sphere = Marker()
    midpoint_sphere.action = Marker.ADD  # create or modify
    midpoint_sphere.type = Marker.SPHERE
    midpoint_sphere.header.frame_id = frame_id
    midpoint_sphere.header.stamp = rospy.Time.now()
    midpoint_sphere.ns = label
    midpoint_sphere.id = 6 * idx + 5
    midpoint_sphere.scale.x = 0.012
    midpoint_sphere.scale.y = 0.012
    midpoint_sphere.scale.z = 0.012
    rope_midpoint = rope_points[int(FloatingRopeScenario.n_links / 2)]
    midpoint_sphere.pose.position.x = rope_midpoint[0]
    midpoint_sphere.pose.position.y = rope_midpoint[1]
    midpoint_sphere.pose.position.z = rope_midpoint[2]
    midpoint_sphere.pose.orientation.x = 0
    midpoint_sphere.pose.orientation.y = 0
    midpoint_sphere.pose.orientation.z = 0
    midpoint_sphere.pose.orientation.w = 1
    midpoint_sphere.color.r = r * 0.8
    midpoint_sphere.color.g = g * 0.8
    midpoint_sphere.color.b = b * 0.8
    midpoint_sphere.color.a = a
    first_point_text = Marker()
    first_point_text.action = Marker.ADD  # create or modify
    first_point_text.type = Marker.TEXT_VIEW_FACING
    first_point_text.header.frame_id = frame_id
    first_point_text.header.stamp = rospy.Time.now()
    first_point_text.ns = label
    first_point_text.id = 6 * idx + 4
    first_point_text.text = "0"
    first_point_text.scale.z = 0.015
    first_point_text.pose.position.x = rope_points[0, 0]
    first_point_text.pose.position.y = rope_points[0, 1]
    first_point_text.pose.position.z = rope_points[0, 2] + 0.015
    first_point_text.pose.orientation.x = 0
    first_point_text.pose.orientation.y = 0
    first_point_text.pose.orientation.z = 0
    first_point_text.pose.orientation.w = 1
    first_point_text.color.r = 1.0
    first_point_text.color.g = 1.0
    first_point_text.color.b = 1.0
    first_point_text.color.a = 1.0
    return [points_marker, lines, midpoint_sphere, first_point_text]


def make_box_marker_from_extents(extent):
    m = Marker()
    ysize, xsize, zsize = extent_to_env_size(extent)
    xcenter, ycenter, zcenter = extent_to_center(extent)
    m.scale.x = xsize
    m.scale.y = ysize
    m.scale.z = zsize
    m.action = Marker.ADD
    m.type = Marker.CUBE
    m.pose.position.x = xcenter
    m.pose.position.y = ycenter
    m.pose.position.z = zcenter
    m.pose.orientation.w = 1
    return m


def sample_rope_and_grippers(rng, g1, g2, p, n_links, kd):
    g1 = g1 + rng.uniform(-0.05, 0.05, 3)
    g2 = g2 + rng.uniform(-0.05, 0.05, 3)
    p = p + rng.uniform(-0.05, 0.05, 3)
    n_exclude = 5
    k = rng.randint(n_exclude, n_links + 1 - n_exclude)
    rope = [g2]
    for i in range(1, k - 1):
        new_p = (p - g2) * (i / (k - 1))
        noise = rng.uniform([-kd, -kd, -kd], [kd, kd, kd], 3)
        new_p = g2 + new_p + noise
        rope.append(new_p)
    rope.append(p)
    for i in range(1, n_links - k + 1):
        new_p = (g1 - p) * i / (n_links - k)
        noise = rng.uniform([-kd, -kd, -kd], [kd, kd, kd], 3)
        new_p = p + new_p + noise
        rope.append(new_p)
    rope = np.array(rope)
    return rope


def sample_rope_grippers(rng, g1, g2, n_links):
    rope = [g2 + rng.uniform(-0.01, 0.01, 3)]
    for _ in range(n_links - 2):
        xmin = min(g1[0], g2[0]) - 0.1
        ymin = min(g1[1], g2[1]) - 0.1
        zmin = min(g1[2], g2[2]) - 0.4
        xmax = max(g1[0], g2[0]) + 0.1
        ymax = max(g1[1], g2[1]) + 0.1
        zmax = max(g1[2], g2[2]) + 0.01
        p = rng.uniform([xmin, ymin, zmin], [xmax, ymax, zmax])
        rope.append(p)
    rope.append(g1 + rng.uniform(-0.01, 0.01, 3))
    return np.array(rope)


def sample_rope(rng, p, n_links, kd: float):
    p = np.array(p, dtype=np.float32)
    n_exclude = 5
    k = rng.randint(n_exclude, n_links + 1 - n_exclude)
    # the kth point of the rope is put at the point p
    rope = [p]
    previous_point = np.copy(p)
    for i in range(0, k):
        noise = rng.uniform([-kd, -kd, -kd / 2], [kd, kd, kd * 1.2], 3)
        previous_point = previous_point + noise
        rope.insert(0, previous_point)
    next_point = np.copy(p)
    for i in range(k, n_links):
        noise = rng.uniform([-kd, -kd, -kd / 2], [kd, kd, kd * 1.2], 3)
        next_point = next_point + noise
        rope.append(next_point)
    rope = np.array(rope)
    return rope


class FloatingRopeScenario(Base3DScenario):
    IMAGE_H = 90
    IMAGE_W = 160
    n_links = 25
    COLOR_IMAGE_TOPIC = "/kinect2/qhd/image_color_rect"
    DEPTH_IMAGE_TOPIC = "/kinect2/qhd/image_depth_rect"
    crop_region = {
        'min_y': 0,
        'min_x': 0,
        'max_y': 540,
        'max_x': 960,
    }
    ROPE_NAMESPACE = 'kinematic_rope'

    # TODO: break out the different pieces of get_state to make them composable,
    #  since there are just a few shared amongst all the scenarios
    # TODO: about this... maybe they should all be pure functions? do we really need "self" at all?
    #  the one reason we have classes at all is so that we can describe interfaces via type hints
    def __init__(self):
        super().__init__()
        self.color_image_listener = Listener(self.COLOR_IMAGE_TOPIC, Image)
        self.depth_image_listener = Listener(self.DEPTH_IMAGE_TOPIC, Image)
        self.state_color_viz_pub = rospy.Publisher("state_color_viz", Image, queue_size=10, latch=True)
        self.state_depth_viz_pub = rospy.Publisher("state_depth_viz", Image, queue_size=10, latch=True)
        self.last_action = None
        self.action_srv = rospy.ServiceProxy(ns_join(self.ROPE_NAMESPACE, "execute_dual_gripper_action"),
                                             GrippersTrajectory)
        self.get_rope_end_points_srv = rospy.ServiceProxy(ns_join(self.ROPE_NAMESPACE, "get_dual_gripper_points"),
                                                          GetDualGripperPoints)
        self.get_rope_srv = rospy.ServiceProxy(ns_join(self.ROPE_NAMESPACE, "get_rope_state"), GetRopeState)
        self.cdcpd_listener = Listener("cdcpd/output", PointCloud2)
        self.set_rope_state_srv = rospy.ServiceProxy(ns_join(self.ROPE_NAMESPACE, "set_rope_state"), SetRopeState)
        self.reset_srv = rospy.ServiceProxy("/gazebo/reset_simulation", Empty)
        self.left_gripper_bbox_pub = rospy.Publisher('/left_gripper_bbox_pub', BoundingBox, queue_size=10, latch=True)
        self.right_gripper_bbox_pub = rospy.Publisher('/right_gripper_bbox_pub', BoundingBox, queue_size=10, latch=True)
        self.overstretching_srv = rospy.ServiceProxy(ns_join(self.ROPE_NAMESPACE, "rope_overstretched"),
                                                     GetOverstretching)
        self.stdev_pub = rospy.Publisher("stdev", Float32, queue_size=10)
        self.error_pub = rospy.Publisher("error", Float32, queue_size=10)

        self.max_action_attempts = 500

        self.robot_reset_rng = np.random.RandomState(0)

    def needs_reset(self):
        res: GetOverstretchingResponse = self.overstretching_srv(GetOverstretchingRequest())
        return res.magnitude > 1.30

    def trajopt_distance_to_goal_differentiable(self, final_state, goal_state: Dict):
        return self.cfm_distance(final_state['z'], goal_state['z'])

    def trajopt_distance_differentiable(self, s1, s2):
        return self.cfm_distance(s1['z'], s2['z'])

    def cfm_distance(self, z1, z2):
        return tf.math.reduce_sum(tf.math.square(z1 - z2), axis=-1, keepdims=True)

    def get_environment(self, params: Dict, **kwargs):
        return {}

    def hard_reset(self):
        self.reset_srv(EmptyRequest())

    def randomization_initialization(self):
        pass

    def on_before_data_collection(self, params: Dict):
        left_gripper_position = np.array([1.0, 0.2, 1.0])
        right_gripper_position = np.array([1.0, -0.2, 1.0])
        init_action = {
            'left_gripper_position': left_gripper_position,
            'right_gripper_position': right_gripper_position,
        }
        self.execute_action(init_action)

    def execute_action(self, action: Dict):
        req = GrippersTrajectoryRequest()
        left_gripper_point = ros_numpy.msgify(Point, action['left_gripper_position'])
        left_gripper_points = Points()
        left_gripper_points.points.append(left_gripper_point)
        req.grippers.append(left_gripper_points)

        right_gripper_point = ros_numpy.msgify(Point, action['right_gripper_position'])
        right_gripper_points = Points()
        right_gripper_points.points.append(right_gripper_point)
        req.grippers.append(right_gripper_points)
        self.action_srv(req)

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

    def sample_action(self,
                      action_rng: np.random.RandomState,
                      environment: Dict,
                      state: Dict,
                      action_params: Dict,
                      stateless: Optional[bool] = False):
        self.viz_action_sample_bbox(self.left_gripper_bbox_pub, self.get_action_sample_extent(action_params, 'left'))
        self.viz_action_sample_bbox(self.right_gripper_bbox_pub, self.get_action_sample_extent(action_params, 'right'))

        action = None
        for _ in range(self.max_action_attempts):
            # move in the same direction as the previous action with some probability
            repeat_probability = action_params['repeat_delta_gripper_motion_probability']
            if not stateless and self.last_action is not None and action_rng.uniform(0, 1) < repeat_probability:
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
                'left_gripper_position': left_gripper_position,
                'right_gripper_position': right_gripper_position,
                'left_gripper_delta_position': left_gripper_delta_position,
                'right_gripper_delta_position': right_gripper_delta_position,
            }

            if self.is_action_valid(action, action_params):
                self.last_action = action
                return action

        rospy.logwarn("Could not find a valid action, executing an invalid one")
        return action

    def is_action_valid(self, action: Dict, action_params: Dict):
        out_of_bounds = FloatingRopeScenario.grippers_out_of_bounds(action['left_gripper_position'],
                                                                    action['right_gripper_position'],
                                                                    action_params)

        max_gripper_d = default_if_none(action_params['max_distance_between_grippers'], 1000)
        too_far = np.linalg.norm(action['left_gripper_position'] - action['right_gripper_position']) > max_gripper_d

        return not out_of_bounds and not too_far

    def get_action_sample_extent(self, action_params: Dict, prefix: str):
        k = prefix + 'gripper_action_sample_extent'
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
                'left_gripper_position': left_gripper_i,
                'right_gripper_position': right_gripper_i,
            }
            interpolated_actions.append(action)

        return interpolated_actions

    @staticmethod
    def robot_name():
        return "rope_3d"

    def randomize_environment(self, env_rng, objects_params: Dict, action_params: Dict):
        pass

    @staticmethod
    def put_state_in_robot_frame(state: Dict):
        rope = state['rope']
        rope_points_shape = rope.shape[:-1].as_list() + [-1, 3]
        rope_points = tf.reshape(rope, rope_points_shape)

        # This assumes robot is at 0 0 0
        robot_position = tf.constant([[0, 0, 0]], tf.float32)
        left_gripper_robot = state['left_gripper']
        right_gripper_robot = state['right_gripper']

        rope_points_robot = rope_points - tf.expand_dims(robot_position, axis=-2)
        rope_robot = tf.reshape(rope_points_robot, rope.shape)

        return {
            'left_gripper': left_gripper_robot,
            'right_gripper': right_gripper_robot,
            'rope': rope_robot,
        }

    @staticmethod
    def put_state_local_frame(state: Dict):
        rope = state['rope']
        rope_points_shape = rope.shape[:-1].as_list() + [-1, 3]
        rope_points = tf.reshape(rope, rope_points_shape)

        center = tf.reduce_mean(rope_points, axis=-2)

        left_gripper_local = state['left_gripper'] - center
        right_gripper_local = state['right_gripper'] - center

        rope_points_local = rope_points - tf.expand_dims(center, axis=-2)
        rope_local = tf.reshape(rope_points_local, rope.shape)

        return {
            'left_gripper': left_gripper_local,
            'right_gripper': right_gripper_local,
            'rope': rope_local,
        }

    @staticmethod
    def local_environment_center_differentiable(state):
        rope_vector = state['rope']
        rope_points = tf.reshape(rope_vector, [rope_vector.shape[0], -1, 3])
        center = tf.reduce_mean(rope_points, axis=1)
        return center

    @staticmethod
    def apply_local_action_at_state(state, local_action):
        return {
            'left_gripper_position': state['left_gripper'] + local_action['left_gripper_delta'],
            'right_gripper_position': state['right_gripper'] + local_action['right_gripper_delta']
        }

    @staticmethod
    def add_noise(action: Dict, noise_rng: np.random.RandomState):
        left_gripper_noise = noise_rng.normal(scale=0.01, size=[3])
        right_gripper_noise = noise_rng.normal(scale=0.01, size=[3])
        return {
            'left_gripper_position': action['left_gripper_position'] + left_gripper_noise,
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
            'left_gripper_delta': left_gripper_delta,
            'right_gripper_delta': right_gripper_delta,
        }

    @staticmethod
    def state_to_gripper_position(state: Dict):
        gripper_position1 = np.reshape(state['left_gripper'], [3])
        gripper_position2 = np.reshape(state['right_gripper'], [3])
        return gripper_position1, gripper_position2

    def get_cdcpd_state(self):
        cdcpd_msg: PointCloud2 = self.cdcpd_listener.get()

        # transform into robot-frame
        transform = self.tf.get_transform_msg("robot_root", "kinect2_rgb_optical_frame")
        cdcpd_points_robot_frame = tf2_sensor_msgs.do_transform_cloud(cdcpd_msg, transform)

        cdcpd_points_array = ros_numpy.numpify(cdcpd_points_robot_frame)
        x = cdcpd_points_array['x']
        y = cdcpd_points_array['y']
        z = cdcpd_points_array['z']
        points = np.stack([x, y, z], axis=-1)

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

    def get_rope_point_positions(self):
        # NOTE: consider getting rid of this message type/service just use rope state [0] and rope state [-1]
        #  although that looses semantic meaning and means hard-coding indices a lot...
        req = GetDualGripperPointsRequest()
        res: GetDualGripperPointsResponse = self.get_rope_end_points_srv(req)
        left_rope_point_position = ros_numpy.numpify(res.left_gripper)
        right_rope_point_position = ros_numpy.numpify(res.right_gripper)
        return left_rope_point_position, right_rope_point_position

    def get_state(self):
        color_depth_cropped = self.get_rgbd()

        rope_state_vector = self.get_rope_state()
        cdcpd_vector = self.get_cdcpd_state()
        left_rope_point_position, right_rope_point_position = self.get_rope_point_positions()

        return {
            'left_gripper': left_rope_point_position,
            'right_gripper': right_rope_point_position,
            'gt_rope': np.array(rope_state_vector, np.float32),
            'rope': np.array(cdcpd_vector, np.float32),
            'rgbd': color_depth_cropped,
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
        box = tf.convert_to_tensor([self.crop_region['min_y'] / rgb.shape[0],
                                    self.crop_region['min_x'] / rgb.shape[1],
                                    self.crop_region['max_y'] / rgb.shape[0],
                                    self.crop_region['max_x'] / rgb.shape[1]], dtype=tf.float32)
        # this operates on a batch
        rgbd_cropped = tf.image.crop_and_resize(image=tf.expand_dims(rgbd, axis=0),
                                                boxes=tf.expand_dims(box, axis=0),
                                                box_indices=[0],
                                                crop_size=[self.IMAGE_H, self.IMAGE_W])
        rgbd_cropped = remove_batch(rgbd_cropped)

        def _debug_show_image(_rgb_depth_cropped):
            import matplotlib.pyplot as plt
            plt.imshow(tf.cast(_rgb_depth_cropped[:, :, :3], tf.int32))
            plt.show()

        # BEGIN DEBUG
        # _debug_show_image(rgbd_cropped)
        # END DEBUG
        return rgbd_cropped.numpy()

    def observations_description(self) -> Dict:
        return {
            'left_gripper': 3,
            'right_gripper': 3,
            'rgbd': [self.IMAGE_H, self.IMAGE_W, 4],
        }

    @staticmethod
    def states_description() -> Dict:
        return {
        }

    @staticmethod
    def observation_features_description() -> Dict:
        return {
            'rope': FloatingRopeScenario.n_links * 3,
            'cdcpd': FloatingRopeScenario.n_links * 3,
        }

    @staticmethod
    def actions_description() -> Dict:
        # should match the keys of the dict return from action_to_dataset_action
        return {
            'left_gripper_position': 3,
            'right_gripper_position': 3,
        }

    @staticmethod
    def state_to_points_for_cc(state: Dict):
        return state['rope'].reshape(-1, 3)

    def __repr__(self):
        return "DualFloatingGripperRope"

    def simple_name(self):
        return "dual_floating"

    @staticmethod
    def sample_gripper_goal(environment: Dict, rng: np.random.RandomState, planner_params: Dict):
        env_inflated = inflate_tf_3d(env=environment['env'],
                                     radius_m=planner_params['goal_threshold'], res=environment['res'])
        goal_extent = planner_params['goal_extent']

        while True:
            extent = np.array(goal_extent).reshape(3, 2)
            left_gripper = rng.uniform(extent[:, 0], extent[:, 1])
            right_gripper = rng.uniform(extent[:, 0], extent[:, 1])
            goal = {
                'left_gripper': left_gripper,
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
        goal_type = planner_params['goal_type']
        if goal_type == 'midpoint':
            return self.sample_midpoint_goal(environment, rng, planner_params)
        elif goal_type == 'rgbd':
            return self.sample_rgbd_goal(environment, rng, planner_params)
        else:
            raise NotImplementedError(planner_params['goal_type'])

    @staticmethod
    def distance_to_gripper_goal(state: Dict, goal: Dict):
        left_gripper = state['left_gripper']
        right_gripper = state['right_gripper']
        distance1 = np.linalg.norm(goal['left_gripper'] - left_gripper)
        distance2 = np.linalg.norm(goal['right_gripper'] - right_gripper)
        return max(distance1, distance2)

    def sample_midpoint_goal(self, environment: Dict, rng: np.random.RandomState, planner_params: Dict):
        goal_extent = planner_params['goal_extent']

        if environment == {}:
            rospy.loginfo("Assuming no obstacles in the environment")
            extent = np.array(goal_extent).reshape(3, 2)
            p = rng.uniform(extent[:, 0], extent[:, 1])
            goal = {'midpoint': p}
            return goal

        env_inflated = inflate_tf_3d(env=environment['env'],
                                     radius_m=planner_params['goal_threshold'], res=environment['res'])
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
    def distance_grippers_and_any_point_goal(state: Dict, goal: Dict):
        rope_points = np.reshape(state['rope'], [-1, 3])
        # well ok not _any_ node, but ones near the middle
        n_from_ends = 5
        distances = np.linalg.norm(np.expand_dims(goal['point'], axis=0) -
                                   rope_points, axis=1)[n_from_ends:-n_from_ends]
        rope_distance = np.min(distances)

        left_gripper = state['left_gripper']
        right_gripper = state['right_gripper']
        distance1 = np.linalg.norm(goal['left_gripper'] - left_gripper)
        distance2 = np.linalg.norm(goal['right_gripper'] - right_gripper)
        return max(max(distance1, distance2), rope_distance)

    @staticmethod
    def distance_to_any_point_goal(state: Dict, goal: Dict):
        rope_points = np.reshape(state['rope'], [-1, 3])
        # well ok not _any_ node, but ones near the middle
        n_from_ends = 7
        distances = np.linalg.norm(np.expand_dims(goal['point'], axis=0) -
                                   rope_points, axis=1)[n_from_ends:-n_from_ends]
        min_distance = np.min(distances)
        return min_distance

    @staticmethod
    def distance_to_midpoint_goal(state: Dict, goal: Dict):
        rope_points = np.reshape(state['rope'], [-1, 3])
        rope_midpoint = rope_points[int(FloatingRopeScenario.n_links / 2)]
        distance = np.linalg.norm(goal['midpoint'] - rope_midpoint)
        return distance

    def classifier_distance(self, s1: Dict, s2: Dict):
        labeling_states = s1['rope']
        labeling_predicted_states = s2['rope']
        points_shape = labeling_states.shape.as_list()[:2] + [-1, 3]
        labeling_points = tf.reshape(labeling_states, points_shape)
        labeling_predicted_points = tf.reshape(labeling_predicted_states, points_shape)
        model_error = tf.reduce_mean(tf.linalg.norm(labeling_points - labeling_predicted_points, axis=-1), axis=-1)
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

    def distance_to_goal(self, state, goal):
        if 'type' not in goal or goal['type'] == 'midpoint':
            return self.distance_to_midpoint_goal(state, goal)
        elif goal['type'] == 'any_point':
            return self.distance_to_any_point_goal(state, goal)
        elif goal['type'] == 'grippers':
            return self.distance_to_gripper_goal(state, goal)
        elif goal['type'] == 'grippers_and_point':
            return self.distance_grippers_and_any_point_goal(state, goal)
        else:
            raise NotImplementedError()

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
        idx1 = self.tree_action_idx * 2 + 0
        idx2 = self.tree_action_idx * 2 + 1
        self.plot_action_rviz(state, action, label='tree', color=[r, g, b, a], idx1=idx1, idx2=idx2, **kwargs)
        self.tree_action_idx += 1

    def plot_executed_action(self, state: Dict, action: Dict, **kwargs):
        self.plot_action_rviz(state, action, label='executed action', color="#3876EB", idx1=1, idx2=1, **kwargs)

    def plot_state_rviz(self, state: Dict, label: str, **kwargs):
        r, g, b, a = colors.to_rgba(kwargs.get("color", "r"))
        idx = kwargs.get("idx", 0)

        msg = MarkerArray()

        if 'gt_rope' in state:
            rope_points = np.reshape(state['gt_rope'], [-1, 3])
            markers = make_rope_marker(rope_points, 'world', label + "_gt_rope", idx, r, g, b, a)
            msg.markers.extend(markers)

        if 'rope' in state:
            rope_points = np.reshape(state['rope'], [-1, 3])
            markers = make_rope_marker(rope_points, 'world', label + "_rope", 1000 + idx, r, g, b, a)
            msg.markers.extend(markers)

        if add_predicted('rope') in state:
            rope_points = np.reshape(state[add_predicted('rope')], [-1, 3])
            markers = make_rope_marker(rope_points, 'world', label + "_pred_rope", 1000 + idx, r, g, b, a,
                                       Marker.CUBE_LIST)
            msg.markers.extend(markers)

        if 'left_gripper' in state:
            r = 0.2
            g = 0.2
            b = 0.8
            left_gripper_sphere = make_gripper_marker(state['left_gripper'], 6 * idx + 2, r, g, b, a, label + '_l',
                                                      Marker.SPHERE)
            msg.markers.append(left_gripper_sphere)

        if 'right_gripper' in state:
            r = 0.8
            g = 0.8
            b = 0.2
            right_gripper_sphere = make_gripper_marker(state['right_gripper'], 6 * idx + 3, r, g, b, a, label + "_r",
                                                       Marker.SPHERE)
            msg.markers.append(right_gripper_sphere)

        if add_predicted('left_gripper') in state:
            r = 0.2
            g = 0.2
            b = 0.8
            lgpp = state[add_predicted('left_gripper')]
            left_gripper_sphere = make_gripper_marker(lgpp, 6 * idx + 2, r, g, b, a, label + "_lp", Marker.CUBE)
            msg.markers.append(left_gripper_sphere)

        if add_predicted('right_gripper') in state:
            r = 0.8
            g = 0.8
            b = 0.2
            rgpp = state[add_predicted('right_gripper')]
            right_gripper_sphere = make_gripper_marker(rgpp, 6 * idx + 3, r, g, b, a, label + "_rp",
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
            stdev_msg = Float32()
            stdev_msg.data = stdev_t
            self.stdev_pub.publish(stdev_msg)

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
        if idx is not None:
            idx1 = idx
            idx2 = idx + 100
        else:
            idx1 = kwargs.pop("idx1", 0)
            idx2 = kwargs.pop("idx2", 1)

        msg = MarkerArray()
        msg.markers.append(rviz_arrow(s1, a1, r, g, b, a, idx=idx1, label=label, **kwargs))
        msg.markers.append(rviz_arrow(s2, a2, r, g, b, a, idx=idx2, label=label, **kwargs))

        self.action_viz_pub.publish(msg)

    @staticmethod
    def numpy_to_ompl_state1(state_np: Dict, state_out: ob.CompoundState):
        for i in range(3):
            state_out[0][i] = np.float64(state_np['left_gripper'][i])
        for i in range(3):
            state_out[1][i] = np.float64(state_np['right_gripper'][i])
        for i in range(FloatingRopeScenario.n_links * 3):
            state_out[2][i] = np.float64(state_np['rope'][i])
        state_out[3][0] = np.float64(state_np['stdev'][0])
        state_out[4][0] = np.float64(state_np['num_diverged'][0])

    @staticmethod
    def numpy_to_ompl_state(state_np: Dict, state_out: ob.CompoundState):
        rope_points = np.reshape(state_np['rope'], [-1, 3])
        for i in range(3):
            state_out[0][i] = np.float64(state_np['left_gripper'][i])
        for i in range(3):
            state_out[1][i] = np.float64(state_np['right_gripper'][i])
        for j in range(FloatingRopeScenario.n_links):
            for i in range(3):
                state_out[2 + j][i] = np.float64(rope_points[j][i])
        state_out[FloatingRopeScenario.n_links + 2][0] = np.float64(state_np['stdev'][0])
        state_out[FloatingRopeScenario.n_links + 3][0] = np.float64(state_np['num_diverged'][0])

    @staticmethod
    def ompl_state_to_numpy1(ompl_state: ob.CompoundState):
        left_gripper = np.array([ompl_state[0][0], ompl_state[0][1], ompl_state[0][2]])
        right_gripper = np.array([ompl_state[1][0], ompl_state[1][1], ompl_state[1][2]])
        rope = []
        for i in range(FloatingRopeScenario.n_links):
            rope.append(ompl_state[2][3 * i + 0])
            rope.append(ompl_state[2][3 * i + 1])
            rope.append(ompl_state[2][3 * i + 2])
        rope = np.array(rope)
        return {
            'left_gripper': left_gripper,
            'right_gripper': right_gripper,
            'rope': rope,
            'stdev': np.array([ompl_state[3][0]]),
            'num_diverged': np.array([ompl_state[4][0]]),
        }

    @staticmethod
    def ompl_state_to_numpy(ompl_state: ob.CompoundState):
        left_gripper = np.array([ompl_state[0][0], ompl_state[0][1], ompl_state[0][2]])
        right_gripper = np.array([ompl_state[1][0], ompl_state[1][1], ompl_state[1][2]])
        rope = []
        for i in range(FloatingRopeScenario.n_links):
            rope.append(ompl_state[2 + i][0])
            rope.append(ompl_state[2 + i][1])
            rope.append(ompl_state[2 + i][2])
        rope = np.array(rope)
        return {
            'left_gripper': left_gripper,
            'right_gripper': right_gripper,
            'rope': rope,
            'stdev': np.array([ompl_state[FloatingRopeScenario.n_links + 2][0]]),
            'num_diverged': np.array([ompl_state[FloatingRopeScenario.n_links + 3][0]]),
        }

    @staticmethod
    def ompl_control_to_numpy(ompl_state: ob.CompoundState, ompl_control: oc.CompoundControl):
        state_np = FloatingRopeScenario.ompl_state_to_numpy(ompl_state)
        current_left_gripper_position = state_np['left_gripper']
        current_right_gripper_position = state_np['right_gripper']

        rotation_matrix_1 = transformations.euler_matrix(0, ompl_control[0][0], ompl_control[0][1])
        left_gripper_delta_position_homo = rotation_matrix_1 @ np.array([1, 0, 0, 1]) * ompl_control[0][2]
        left_gripper_delta_position = left_gripper_delta_position_homo[:3]

        rotation_matrix_2 = transformations.euler_matrix(0, ompl_control[1][0], ompl_control[1][1])
        right_gripper_delta_position_homo = rotation_matrix_2 @ np.array([1, 0, 0, 1]) * ompl_control[1][2]
        right_gripper_delta_position = right_gripper_delta_position_homo[:3]

        target_left_gripper_position = current_left_gripper_position + left_gripper_delta_position
        target_right_gripper_position = current_right_gripper_position + right_gripper_delta_position
        return {
            'left_gripper_position': target_left_gripper_position,
            'right_gripper_position': target_right_gripper_position,
        }

    def make_goal_region(self, si: oc.SpaceInformation, rng: np.random.RandomState, params: Dict, goal: Dict,
                         plot: bool):
        if 'type' not in goal or goal['type'] == 'midpoint':
            return RopeMidpointGoalRegion(si=si,
                                          scenario=self,
                                          rng=rng,
                                          threshold=params['goal_threshold'],
                                          goal=goal,
                                          plot=plot)
        elif goal['type'] == 'any_point':
            return RopeAnyPointGoalRegion(si=si,
                                          scenario=self,
                                          rng=rng,
                                          threshold=params['goal_threshold'],
                                          goal=goal,
                                          plot=plot)
        elif goal['type'] == 'grippers':
            return DualGripperGoalRegion(si=si,
                                         scenario=self,
                                         rng=rng,
                                         threshold=params['goal_threshold'],
                                         goal=goal,
                                         plot=plot)
        elif goal['type'] == 'grippers_and_point':
            return RopeAndGrippersGoalRegion(si=si,
                                             scenario=self,
                                             rng=rng,
                                             threshold=params['goal_threshold'],
                                             goal=goal,
                                             plot=plot)
        else:
            raise NotImplementedError()

    def make_ompl_state_space(self, planner_params, state_sampler_rng: np.random.RandomState, plot: bool):
        state_space = ob.CompoundStateSpace()

        min_x, max_x, min_y, max_y, min_z, max_z = planner_params['extent']

        left_gripper_subspace = ob.RealVectorStateSpace(3)
        left_gripper_bounds = ob.RealVectorBounds(3)
        left_gripper_bounds.setLow(0, min_x)
        left_gripper_bounds.setHigh(0, max_x)
        left_gripper_bounds.setLow(1, min_y)
        left_gripper_bounds.setHigh(1, max_y)
        left_gripper_bounds.setLow(2, min_z)
        left_gripper_bounds.setHigh(2, max_z)
        left_gripper_subspace.setBounds(left_gripper_bounds)
        left_gripper_subspace.setName("left_gripper")
        state_space.addSubspace(left_gripper_subspace, weight=1)

        right_gripper_subspace = ob.RealVectorStateSpace(3)
        right_gripper_bounds = ob.RealVectorBounds(3)
        right_gripper_bounds.setLow(0, min_x)
        right_gripper_bounds.setHigh(0, max_x)
        right_gripper_bounds.setLow(1, min_y)
        right_gripper_bounds.setHigh(1, max_y)
        right_gripper_bounds.setLow(2, min_z)
        right_gripper_bounds.setHigh(2, max_z)
        right_gripper_subspace.setBounds(right_gripper_bounds)
        right_gripper_subspace.setName("right_gripper")
        state_space.addSubspace(right_gripper_subspace, weight=1)

        for i in range(FloatingRopeScenario.n_links):
            rope_point_subspace = ob.RealVectorStateSpace(3)
            rope_point_bounds = ob.RealVectorBounds(3)
            rope_point_bounds.setLow(0, min_x)
            rope_point_bounds.setHigh(0, max_x)
            rope_point_bounds.setLow(1, min_y)
            rope_point_bounds.setHigh(1, max_y)
            rope_point_bounds.setLow(2, min_z)
            rope_point_bounds.setHigh(2, max_z)
            rope_point_subspace.setBounds(rope_point_bounds)
            rope_point_subspace.setName(f"rope_{i}")
            state_space.addSubspace(rope_point_subspace, weight=1)

        # extra subspace component for the variance, which is necessary to pass information from propagate to constraint checker
        stdev_subspace = ob.RealVectorStateSpace(1)
        stdev_bounds = ob.RealVectorBounds(1)
        stdev_bounds.setLow(-1000)
        stdev_bounds.setHigh(1000)
        stdev_subspace.setBounds(stdev_bounds)
        stdev_subspace.setName("stdev")
        state_space.addSubspace(stdev_subspace, weight=0)

        # extra subspace component for the number of diverged steps
        num_diverged_subspace = ob.RealVectorStateSpace(1)
        num_diverged_bounds = ob.RealVectorBounds(1)
        num_diverged_bounds.setLow(-1000)
        num_diverged_bounds.setHigh(1000)
        num_diverged_subspace.setBounds(num_diverged_bounds)
        num_diverged_subspace.setName("stdev")
        state_space.addSubspace(num_diverged_subspace, weight=0)

        def _state_sampler_allocator(state_space):
            return DualGripperStateSampler(state_space,
                                           scenario=self,
                                           extent=planner_params['state_sampler_extent'],
                                           rng=state_sampler_rng,
                                           plot=plot)

        state_space.setStateSamplerAllocator(ob.StateSamplerAllocator(_state_sampler_allocator))

        return state_space

    def make_ompl_control_space(self, state_space, rng: np.random.RandomState, action_params: Dict):
        control_space = oc.CompoundControlSpace(state_space)

        left_gripper_control_space = oc.RealVectorControlSpace(state_space, 3)
        left_gripper_control_bounds = ob.RealVectorBounds(3)
        # Pitch
        left_gripper_control_bounds.setLow(0, -np.pi)
        left_gripper_control_bounds.setHigh(0, np.pi)
        # Yaw
        left_gripper_control_bounds.setLow(1, -np.pi)
        left_gripper_control_bounds.setHigh(1, np.pi)
        # Displacement
        max_d = action_params['max_distance_gripper_can_move']
        left_gripper_control_bounds.setLow(2, 0)
        left_gripper_control_bounds.setHigh(2, max_d)
        left_gripper_control_space.setBounds(left_gripper_control_bounds)
        control_space.addSubspace(left_gripper_control_space)

        right_gripper_control_space = oc.RealVectorControlSpace(state_space, 3)
        right_gripper_control_bounds = ob.RealVectorBounds(3)
        # Pitch
        right_gripper_control_bounds.setLow(0, -np.pi)
        right_gripper_control_bounds.setHigh(0, np.pi)
        # Yaw
        right_gripper_control_bounds.setLow(1, -np.pi)
        right_gripper_control_bounds.setHigh(1, np.pi)
        # Displacement
        max_d = action_params['max_distance_gripper_can_move']
        right_gripper_control_bounds.setLow(2, 0)
        right_gripper_control_bounds.setHigh(2, max_d)

        right_gripper_control_space.setBounds(right_gripper_control_bounds)
        control_space.addSubspace(right_gripper_control_space)

        def _allocator(cs):
            return DualGripperControlSampler(cs, scenario=self, rng=rng, action_params=action_params)

        # I override the sampler here so I can use numpy RNG to make things more deterministic.
        # ompl does not allow resetting of seeds, which causes problems when evaluating multiple
        # planning queries in a row.
        control_space.setControlSamplerAllocator(oc.ControlSamplerAllocator(_allocator))

        return control_space


class DualGripperControlSampler(oc.ControlSampler):
    def __init__(self,
                 control_space: oc.CompoundControlSpace,
                 scenario: FloatingRopeScenario,
                 rng: np.random.RandomState,
                 action_params: Dict):
        super().__init__(control_space)
        self.scenario = scenario
        self.rng = rng
        self.control_space = control_space
        self.action_params = action_params

    def sampleNext(self, control_out, previous_control, state):
        del previous_control
        del state

        # Pitch
        pitch_1 = self.rng.uniform(-np.pi, np.pi)
        pitch_2 = self.rng.uniform(-np.pi, np.pi)
        # Yaw
        yaw_1 = self.rng.uniform(-np.pi, np.pi)
        yaw_2 = self.rng.uniform(-np.pi, np.pi)
        # Displacement
        displacement1 = self.rng.uniform(0, self.action_params['max_distance_gripper_can_move'])
        displacement2 = self.rng.uniform(0, self.action_params['max_distance_gripper_can_move'])

        control_out[0][0] = pitch_1
        control_out[0][1] = yaw_1
        control_out[0][2] = displacement1

        control_out[1][0] = pitch_2
        control_out[1][1] = yaw_2
        control_out[1][2] = displacement2

    def sampleStepCount(self, min_steps, max_steps):
        step_count = self.rng.randint(min_steps, max_steps)
        return step_count


class DualGripperStateSampler(ob.CompoundStateSampler):

    def __init__(self,
                 state_space,
                 scenario: FloatingRopeScenario,
                 extent,
                 rng: np.random.RandomState,
                 plot: bool):
        super().__init__(state_space)
        self.state_space = state_space
        self.scenario = scenario
        self.extent = np.array(extent).reshape(3, 2)
        self.rng = rng
        self.plot = plot

        bbox_msg = extent_to_bbox(extent)
        bbox_msg.header.frame_id = 'world'
        self.sampler_extents_bbox_pub = rospy.Publisher('sampler_extents', BoundingBox, queue_size=10, latch=True)
        self.sampler_extents_bbox_pub.publish(bbox_msg)

    def sample_point_for_R3_subspace(self, subspace, subspace_state_out):
        bounds = subspace.getBounds()
        min_x = bounds.low[0]
        min_y = bounds.low[1]
        min_z = bounds.low[2]
        max_x = bounds.high[0]
        max_y = bounds.high[1]
        max_z = bounds.high[2]
        p = self.rng.uniform([min_x, min_y, min_z], [max_x, max_y, max_z])
        subspace_state_out[0] = p[0]
        subspace_state_out[1] = p[1]
        subspace_state_out[2] = p[2]

    def sampleUniform(self, state_out: ob.CompoundState):
        # for i in range(2 + DualFloatingGripperRopeScenario.n_links):
        #     self.sample_point_for_R3_subspace(self.state_space.getSubspace(i), state_out[i])
        # state_np = self.scenario.ompl_state_to_numpy(state_out)

        random_point = self.rng.uniform(self.extent[:, 0], self.extent[:, 1])
        random_point_rope = np.concatenate([random_point] * FloatingRopeScenario.n_links)
        state_np = {
            'left_gripper': random_point,
            'right_gripper': random_point,
            'rope': random_point_rope,
            'num_diverged': np.zeros(1, dtype=np.float64),
            'stdev': np.zeros(1, dtype=np.float64),
        }
        self.scenario.numpy_to_ompl_state(state_np, state_out)

        if self.plot:
            self.scenario.plot_sampled_state(state_np)


class DualGripperGoalRegion(ob.GoalSampleableRegion):

    def __init__(self,
                 si: oc.SpaceInformation,
                 scenario: FloatingRopeScenario,
                 rng: np.random.RandomState,
                 threshold: float,
                 goal: Dict,
                 plot: bool):
        super(DualGripperGoalRegion, self).__init__(si)
        self.setThreshold(threshold)
        self.goal = goal
        self.scenario = scenario
        self.rng = rng
        self.plot = plot

    def distanceGoal(self, state: ob.CompoundState):
        """
        Uses the distance between a specific point in a specific subspace and the goal point
        """
        state_np = self.scenario.ompl_state_to_numpy(state)
        distance = self.scenario.distance_to_gripper_goal(state_np, self.goal)

        # this ensures the goal must have num_diverged = 0
        if state_np['num_diverged'] > 0:
            distance = 1e9
        return distance

    def sampleGoal(self, state_out: ob.CompoundState):
        sampler = self.getSpaceInformation().allocStateSampler()
        # sample a random state via the state space sampler, in hopes that OMPL will clean up the memory...
        sampler.sampleUniform(state_out)

        # don't bother trying to sample "legit" rope states, because this is only used to bias sampling towards the goal
        # so just prenteing every point on therope is at the goal should be sufficient
        rope = sample_rope_grippers(self.rng,
                                    self.goal['left_gripper'],
                                    self.goal['right_gripper'],
                                    FloatingRopeScenario.n_links)

        goal_state_np = {
            'left_gripper': self.goal['left_gripper'],
            'right_gripper': self.goal['right_gripper'],
            'rope': rope.flatten(),
            'num_diverged': np.zeros(1, dtype=np.float64),
            'stdev': np.zeros(1, dtype=np.float64),
        }

        self.scenario.numpy_to_ompl_state(goal_state_np, state_out)

        if self.plot:
            self.scenario.plot_sampled_goal_state(goal_state_np)

    def maxSampleCount(self):
        return 100


class RopeMidpointGoalRegion(ob.GoalSampleableRegion):

    def __init__(self,
                 si: oc.SpaceInformation,
                 scenario: FloatingRopeScenario,
                 rng: np.random.RandomState,
                 threshold: float,
                 goal: Dict,
                 plot: bool):
        super(RopeMidpointGoalRegion, self).__init__(si)
        self.setThreshold(threshold)
        self.goal = goal
        self.scenario = scenario
        self.rng = rng
        self.plot = plot

    def distanceGoal(self, state: ob.CompoundState):
        """
        Uses the distance between a specific point in a specific subspace and the goal point
        """
        state_np = self.scenario.ompl_state_to_numpy(state)
        distance = self.scenario.distance_to_midpoint_goal(state_np, self.goal)

        # this ensures the goal must have num_diverged = 0
        if state_np['num_diverged'] > 0:
            distance = 1e9
        return distance

    def sampleGoal(self, state_out: ob.CompoundState):
        sampler = self.getSpaceInformation().allocStateSampler()
        # sample a random state via the state space sampler, in hopes that OMPL will clean up the memory...
        sampler.sampleUniform(state_out)

        # attempt to sample "legit" rope states
        kd = 0.04
        rope = sample_rope(self.rng, self.goal['midpoint'], FloatingRopeScenario.n_links, kd)
        # gripper 1 is attached to the last link
        left_gripper = rope[-1] + self.rng.uniform(-kd, kd, 3)
        right_gripper = rope[0] + self.rng.uniform(-kd, kd, 3)

        goal_state_np = {
            'left_gripper': left_gripper,
            'right_gripper': right_gripper,
            'rope': rope.flatten(),
            'num_diverged': np.zeros(1, dtype=np.float64),
            'stdev': np.zeros(1, dtype=np.float64),
        }

        self.scenario.numpy_to_ompl_state(goal_state_np, state_out)

        if self.plot:
            self.scenario.plot_sampled_goal_state(goal_state_np)

    def maxSampleCount(self):
        return 100


class RopeAnyPointGoalRegion(ob.GoalSampleableRegion):

    def __init__(self,
                 si: oc.SpaceInformation,
                 scenario: FloatingRopeScenario,
                 rng: np.random.RandomState,
                 threshold: float,
                 goal: Dict,
                 plot: bool):
        super(RopeAnyPointGoalRegion, self).__init__(si)
        self.setThreshold(threshold)
        self.goal = goal
        self.scenario = scenario
        self.rng = rng
        self.plot = plot

    def distanceGoal(self, state: ob.CompoundState):
        """
        Uses the distance between a specific point in a specific subspace and the goal point
        """
        state_np = self.scenario.ompl_state_to_numpy(state)
        distance = self.scenario.distance_to_any_point_goal(state_np, self.goal)

        # this ensures the goal must have num_diverged = 0
        if state_np['num_diverged'] > 0:
            distance = 1e9
        return distance

    def sampleGoal(self, state_out: ob.CompoundState):
        sampler = self.getSpaceInformation().allocStateSampler()
        # sample a random state via the state space sampler, in hopes that OMPL will clean up the memory...
        sampler.sampleUniform(state_out)

        # attempt to sample "legit" rope states
        kd = 0.05
        rope = sample_rope(self.rng, self.goal['point'], FloatingRopeScenario.n_links, kd)
        # gripper 1 is attached to the last link
        left_gripper = rope[-1] + self.rng.uniform(-kd, kd, 3)
        right_gripper = rope[0] + self.rng.uniform(-kd, kd, 3)

        goal_state_np = {
            'left_gripper': left_gripper,
            'right_gripper': right_gripper,
            'rope': rope.flatten(),
            'num_diverged': np.zeros(1, dtype=np.float64),
            'stdev': np.zeros(1, dtype=np.float64),
        }

        self.scenario.numpy_to_ompl_state(goal_state_np, state_out)

        if self.plot:
            self.scenario.plot_sampled_goal_state(goal_state_np)

    def maxSampleCount(self):
        return 100


class RopeAndGrippersGoalRegion(ob.GoalSampleableRegion):

    def __init__(self,
                 si: oc.SpaceInformation,
                 scenario: FloatingRopeScenario,
                 rng: np.random.RandomState,
                 threshold: float,
                 goal: Dict,
                 plot: bool):
        super(RopeAndGrippersGoalRegion, self).__init__(si)
        self.setThreshold(threshold)
        self.goal = goal
        self.scenario = scenario
        self.rng = rng
        self.plot = plot

    def distanceGoal(self, state: ob.CompoundState):
        state_np = self.scenario.ompl_state_to_numpy(state)
        distance = self.scenario.distance_grippers_and_any_point_goal(state_np, self.goal)

        # this ensures the goal must have num_diverged = 0
        if state_np['num_diverged'] > 0:
            distance = 1e9
        return distance

    def sampleGoal(self, state_out: ob.CompoundState):
        # attempt to sample "legit" rope states
        kd = 0.05
        rope = sample_rope_and_grippers(
            self.rng, self.goal['left_gripper'], self.goal['right_gripper'], self.goal['point'],
            FloatingRopeScenario.n_links,
            kd)

        goal_state_np = {
            'left_gripper': self.goal['left_gripper'],
            'right_gripper': self.goal['right_gripper'],
            'rope': rope.flatten(),
            'num_diverged': np.zeros(1, dtype=np.float64),
            'stdev': np.zeros(1, dtype=np.float64),
        }

        self.scenario.numpy_to_ompl_state(goal_state_np, state_out)

        if self.plot:
            self.scenario.plot_sampled_goal_state(goal_state_np)

    def maxSampleCount(self):
        return 100


class RopeAndGrippersBoxesGoalRegion(ob.GoalSampleableRegion):

    def __init__(self,
                 si: oc.SpaceInformation,
                 scenario: FloatingRopeScenario,
                 rng: np.random.RandomState,
                 threshold: float,
                 goal: Dict,
                 plot: bool):
        super(RopeAndGrippersBoxesGoalRegion, self).__init__(si)
        self.goal = goal
        self.scenario = scenario
        self.setThreshold(threshold)
        self.rng = rng
        self.plot = plot

    def isSatisfied(self, state: ob.CompoundState, distance):
        state_np = self.scenario.ompl_state_to_numpy(state)
        rope_points = np.reshape(state_np['rope'], [-1, 3])
        n_from_ends = 7
        near_center_rope_points = rope_points[n_from_ends:-n_from_ends]

        left_gripper_extent = np.reshape(self.goal['left_gripper_box'], [3, 2])
        left_gripper_satisfied = np.logical_and(
            state_np['left_gripper'] >= left_gripper_extent[:, 0],
            state_np['left_gripper'] <= left_gripper_extent[:, 1])

        right_gripper_extent = np.reshape(self.goal['right_gripper_box'], [3, 2])
        right_gripper_satisfied = np.logical_and(
            state_np['right_gripper'] >= right_gripper_extent[:, 0],
            state_np['right_gripper'] <= right_gripper_extent[:, 1])

        point_extent = np.reshape(self.goal['point_box'], [3, 2])
        points_satisfied = np.logical_and(near_center_rope_points >=
                                          point_extent[:, 0], near_center_rope_points <= point_extent[:, 1])
        any_point_satisfied = np.reduce_any(points_satisfied)

        return float(any_point_satisfied and left_gripper_satisfied and right_gripper_satisfied)

    def sampleGoal(self, state_out: ob.CompoundState):
        # attempt to sample "legit" rope states
        kd = 0.05
        rope = sample_rope_and_grippers(
            self.rng, self.goal['left_gripper'], self.goal['right_gripper'], self.goal['point'],
            FloatingRopeScenario.n_links,
            kd)

        goal_state_np = {
            'left_gripper': self.goal['left_gripper'],
            'right_gripper': self.goal['right_gripper'],
            'rope': rope.flatten(),
            'num_diverged': np.zeros(1, dtype=np.float64),
            'stdev': np.zeros(1, dtype=np.float64),
        }

        self.scenario.numpy_to_ompl_state(goal_state_np, state_out)

        if self.plot:
            self.scenario.plot_sampled_goal_state(goal_state_np)

    def distanceGoal(self, state: ob.CompoundState):
        state_np = self.scenario.ompl_state_to_numpy(state)
        distance = self.scenario.distance_grippers_and_any_point_goal(state_np, self.goal)

        # this ensures the goal must have num_diverged = 0
        if state_np['num_diverged'] > 0:
            distance = 1e9
        return distance

    def maxSampleCount(self):
        return 100
