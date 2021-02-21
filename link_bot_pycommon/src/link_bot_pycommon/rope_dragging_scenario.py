from typing import Dict, Optional

import numpy as np
import tensorflow as tf
from matplotlib import colors

import ros_numpy
import rospy
from geometry_msgs.msg import Point
from link_bot_data.dataset_utils import add_predicted, get_maybe_predicted
from link_bot_data.visualization import rviz_arrow
from link_bot_gazebo_python.gazebo_services import gz_scope
from link_bot_gazebo_python.position_3d import Position3D
from link_bot_pycommon.base_3d_scenario import Base3DScenario
from link_bot_pycommon.base_services import BaseServices
from link_bot_pycommon.collision_checking import inflate_tf_3d
from link_bot_pycommon.get_occupancy import get_environment_for_extents_3d
from link_bot_pycommon.grid_utils import point_to_idx_3d_in_env
from link_bot_pycommon.make_rope_markers import make_rope_marker, make_gripper_marker
from link_bot_pycommon.marker_index_generator import marker_index_generator
from moonshine.base_learned_dynamics_model import dynamics_loss_function, dynamics_points_metrics_function
from peter_msgs.srv import *
from rosgraph.names import ns_join
from std_msgs.msg import ColorRGBA
from std_srvs.srv import Empty
from visualization_msgs.msg import MarkerArray, Marker

rope_key_name = 'rope'
gazebo_model_name = "dragging_rope"


class RopeDraggingScenario(Base3DScenario):
    n_links = 10
    ROPE_NAMESPACE = 'dragging_rope'
    ROPE_LINK_NAME = gz_scope(ROPE_NAMESPACE, 'gripper1')

    def __init__(self):
        super().__init__()
        self.service_provider = BaseServices()

        self.pos3d = Position3D()
        self.get_rope_srv = rospy.ServiceProxy(ns_join(self.ROPE_NAMESPACE, "get_rope_state"), GetRopeState)
        self.reset_sim_srv = rospy.ServiceProxy("/gazebo/reset_simulation", Empty)

        self.last_action = None
        self.max_action_attempts = 1000

    def __repr__(self):
        return "rope_dragging_scenario"

    def plot_state_rviz(self, state: Dict, **kwargs):
        label = kwargs.get("label", "")
        color_msg = ColorRGBA(*colors.to_rgba(kwargs.get("color", "r")))
        idx = kwargs.get("idx", 0)
        ig = marker_index_generator(idx)

        msg = MarkerArray()
        if rope_key_name in state:
            rope_points = np.reshape(state[rope_key_name], [-1, 3])
            rope_mrkrs = make_rope_marker(rope_points, 'world', label + "_gt_" + rope_key_name, next(ig), color_msg,
                                          s=0.04)
            points_marker, lines, midpoint_sphere, first_point_text = rope_mrkrs
            msg.markers.append(lines)
            # msg.markers.append(points_marker)
            # msg.markers.append(first_point_text)

        if 'gripper' in state:
            gripper = state['gripper']
            gripper_sphere = make_gripper_marker(gripper, next(ig), color_msg, label + '_gt_gripper', Marker.SPHERE,
                                                 s=0.04)
            msg.markers.append(gripper_sphere)

        if add_predicted(rope_key_name) in state:
            rope_points = np.reshape(state[add_predicted(rope_key_name)], [-1, 3])
            markers = make_rope_marker(rope_points, 'world', label + "_" + rope_key_name, next(ig), color_msg)
            msg.markers.extend(markers)

        if add_predicted('gripper') in state:
            pred_gripper = state[add_predicted('gripper')]
            gripper_sphere = make_gripper_marker(pred_gripper, next(ig), color_msg, label + "_gripper", Marker.SPHERE)
            msg.markers.append(gripper_sphere)

        self.state_viz_pub.publish(msg)

    def plot_action_rviz(self, state: Dict, action: Dict, label: str = 'action', **kwargs):
        state_action = {}
        state_action.update(state)
        state_action.update(action)
        self.plot_action_rviz_internal(state_action, label=label, **kwargs)

    def plot_action_rviz_internal(self, data: Dict, label: str, **kwargs):
        r, g, b, a = colors.to_rgba(kwargs.get("color", "b"))
        z_offset = 0.05
        gripper = np.reshape(get_maybe_predicted(data, 'gripper'), [3]) + [0, 0, z_offset]
        target_gripper = np.reshape(get_maybe_predicted(data, 'gripper_position'), [3]) + [0, 0, z_offset]

        idx = kwargs.get("idx", 0)

        msg = MarkerArray()
        msg.markers.append(rviz_arrow(position=gripper,
                                      target_position=target_gripper,
                                      r=r, g=g, b=b, a=a,
                                      idx=idx, label=label, **kwargs))

        self.action_viz_pub.publish(msg)

    def on_before_get_state_or_execute_action(self):
        self.pos3d.register(RegisterPosition3DControllerRequest(scoped_link_name=self.ROPE_LINK_NAME,
                                                                controller_type='pid',
                                                                kp_vel=10.0,
                                                                kp_pos=10.0,
                                                                max_force=10.0,
                                                                max_vel=0.1))

    def register_movable_object(self, scoped_link_name):
        self.pos3d.register(RegisterPosition3DControllerRequest(scoped_link_name=scoped_link_name,
                                                                controller_type='pid',
                                                                kp_pos=50.0,
                                                                kp_vel=1000.0,
                                                                max_force=50.0,
                                                                max_vel=1.0))

    def on_before_data_collection(self, params: Dict):
        self.on_before_get_state_or_execute_action()

    def execute_action(self, action: Dict):
        timeout_s = action.get('timeout_s', [1.0])[0]
        speed_mps = action.get('speed', 0.15)
        pos_msg: Point = ros_numpy.msgify(Point, action['gripper_position'])
        get_res = self.pos3d.get(scoped_link_name=self.ROPE_LINK_NAME)
        pos_msg.z = get_res.pos.z
        req = Position3DActionRequest(scoped_link_name=self.ROPE_LINK_NAME,
                                      position=pos_msg,
                                      tolerance_m=0.005,
                                      speed_mps=speed_mps,
                                      timeout_s=timeout_s,
                                      )
        self.pos3d.move(req)

    def sample_action(self,
                      action_rng: np.random.RandomState,
                      environment: Dict,
                      state: Dict,
                      action_params: Dict,
                      validate: bool,
                      stateless: Optional[bool] = False):
        action = None
        for _ in range(self.max_action_attempts):
            # sample the previous action with 80% probability, this improves exploration
            repeat_probability = action_params.get('repeat_delta_gripper_motion_probability', 0.8)
            if self.last_action is not None and action_rng.uniform(0, 1) < repeat_probability and not stateless:
                gripper_delta_position = self.last_action['gripper_delta_position']
            else:
                theta = action_rng.uniform(-np.pi, np.pi)
                displacement = action_rng.uniform(0, action_params['max_distance_gripper_can_move'])

                dx = np.cos(theta) * displacement
                dy = np.sin(theta) * displacement

                gripper_delta_position = np.array([dx, dy, 0])

            gripper_position = state['gripper'] + gripper_delta_position
            action = {
                'gripper_position':       gripper_position,
                'gripper_delta_position': gripper_delta_position,
                'timeout_s':              np.array([action_params['dt']]),
            }

            valid = self.is_action_valid(action, action_params)
            if not validate or valid:
                self.last_action = action
                return action

        rospy.logwarn("Could not find a valid action, executing an invalid one")
        return action

    def is_action_valid(self, action: Dict, action_params: Dict):
        out_of_bounds = self.gripper_out_of_bounds(action['gripper_position'], action_params)
        return not out_of_bounds

    @staticmethod
    def interpolate(start_state, end_state, step_size=0.05):
        gripper_start = np.array(start_state['gripper'])
        gripper_end = np.array(end_state['gripper'])

        gripper_delta = gripper_end - gripper_start

        steps = np.round(np.linalg.norm(gripper_delta) / step_size).astype(np.int64)

        interpolated_actions = []
        for t in np.linspace(step_size, 1, steps):
            gripper_t = gripper_start + gripper_delta * t
            action = {
                'gripper_position': gripper_t,
            }
            interpolated_actions.append(action)

        return interpolated_actions

    @staticmethod
    def add_noise(action: Dict, noise_rng: np.random.RandomState):
        gripper_noise = noise_rng.normal(scale=0.01, size=[3])
        return {
            'gripper_position': action['gripper_position'] + gripper_noise
        }

    def gripper_out_of_bounds(self, gripper, data_collection_params: Dict):
        gripper_extent = data_collection_params['gripper_action_sample_extent']
        return RopeDraggingScenario.is_out_of_bounds(gripper, gripper_extent)

    @staticmethod
    def is_out_of_bounds(p, extent):
        x, y, z = p
        x_min, x_max, y_min, y_max, z_min, z_max = extent
        return x < x_min or x > x_max \
               or y < y_min or y > y_max \
               or z < z_min or z > z_max

    def get_state(self):
        gripper_res = self.pos3d.get(scoped_link_name=self.ROPE_LINK_NAME)
        assert gripper_res.success

        rope_res = self.get_rope_srv(GetRopeStateRequest())

        rope_state_vector = []
        assert (len(rope_res.positions) == RopeDraggingScenario.n_links)
        for p in rope_res.positions:
            rope_state_vector.append(p.x)
            rope_state_vector.append(p.y)
            rope_state_vector.append(p.z)

        return {
            'gripper':     ros_numpy.numpify(gripper_res.pos),
            rope_key_name: np.array(rope_state_vector, np.float32),
        }

    @staticmethod
    def states_description() -> Dict:
        return {
            'gripper':     3,
            rope_key_name: RopeDraggingScenario.n_links * 3,
        }

    @staticmethod
    def actions_description() -> Dict:
        # should match the keys of the dict return from action_to_dataset_action
        return {
            'gripper_position': 3,
            'timeout_s':        1,
        }

    @staticmethod
    def distance_to_goal(
            state: Dict[str, np.ndarray],
            goal: np.ndarray):
        """
        Uses the first point in the rope subspace as the thing which we want to move to goal
        :param state: A dictionary of numpy arrays
        :param goal: Assumed to be a point in 3D
        :return:
        """
        rope_points = np.reshape(state[rope_key_name], [-1, 3])
        tail_point = rope_points[0]
        distance = np.linalg.norm(tail_point - goal['tail'])
        return distance

    @staticmethod
    def distance_to_goal_differentiable(state, goal):
        rope_points = tf.reshape(state[rope_key_name], [-1, 3])
        tail_point = rope_points[0]
        distance = tf.linalg.norm(tail_point - goal)
        return distance

    def classifier_distance(self, s1: Dict, s2: Dict):
        model_error = np.linalg.norm(s1[rope_key_name] - s2[rope_key_name], axis=-1)
        return model_error

    @staticmethod
    def distance_differentiable(s1, s2):
        rope_points1 = tf.reshape(s1[rope_key_name], [-1, 3])
        tail_point1 = rope_points1[0]
        rope_points2 = tf.reshape(s2[rope_key_name], [-1, 3])
        tail_point2 = rope_points2[0]
        return tf.linalg.norm(tail_point1 - tail_point2)

    @staticmethod
    def state_to_points_for_cc(state: Dict):
        return state[rope_key_name].reshape(-1, 3)

    def sample_goal(self, environment: Dict, rng: np.random.RandomState, planner_params: Dict):
        goal_type = planner_params['goal_params']['goal_type']
        if goal_type == 'tailpoint':
            return self.sample_tailpoint_goal(environment, rng, planner_params)
        else:
            raise NotImplementedError(planner_params['goal_type'])

    def sample_tailpoint_goal(self, environment: Dict, rng: np.random.RandomState, planner_params: Dict):
        # add more inflating to reduce the number of truly unacheivable gols
        env_inflated = inflate_tf_3d(env=environment['env'],
                                     radius_m=2 * planner_params['goal_params']['threshold'],
                                     res=environment['res'])
        goal_extent = planner_params['goal_params']['extent']

        while True:
            extent = np.array(goal_extent).reshape(3, 2)
            p = rng.uniform(extent[:, 0], extent[:, 1])
            goal = {'tail': p}
            row, col, channel = point_to_idx_3d_in_env(p[0], p[1], p[2], environment)
            collision = env_inflated[row, col, channel] > 0.5
            if not collision:
                return goal

    @staticmethod
    def local_environment_center_differentiable(state):
        return state['gripper']

    @staticmethod
    def simple_name():
        return "rope dragging"

    @staticmethod
    def dynamics_loss_function(dataset_element, predictions):
        return dynamics_loss_function(dataset_element, predictions)

    @staticmethod
    def dynamics_metrics_function(dataset_element, predictions):
        return dynamics_points_metrics_function(dataset_element, predictions)

    @staticmethod
    def put_state_robot_frame(state: Dict):
        return state

    @staticmethod
    def put_state_local_frame(state):
        rope = state[rope_key_name]
        rope_points_shape = rope.shape[:-1].as_list() + [-1, 3]
        points = tf.reshape(rope, rope_points_shape)
        center = state['gripper']
        rope_points_local = points - tf.expand_dims(center, axis=-2)
        gripper_local = state['gripper'] - center

        # # project all z coordinates down to what we saw in simulation... the real fix for this is to use TF and have actually cordinate frames
        # gripper_local = gripper_local * tf.constant([[1, 1, 0]], tf.float32) + tf.constant([[1, 1, 0.02]], tf.float32)
        # rope_points_local = rope_points_local * tf.constant([[1, 1, 0]], tf.float32)+ tf.constant([[1, 1, 0.02]], tf.float32)

        rope_local = tf.reshape(rope_points_local, rope.shape)

        return {
            'gripper':     gripper_local,
            rope_key_name: rope_local,
        }

    @staticmethod
    def put_action_local_frame(state: Dict, action: Dict):
        target_gripper_position = action['gripper_position']

        current_gripper_point = state['gripper']

        gripper_delta = target_gripper_position - current_gripper_point

        return {
            'gripper_delta': gripper_delta,
        }

    def randomization_initialization(self, params: Dict):
        pass

    def randomize_environment(self, env_rng, params: Dict):
        # TODO: make scenarios take in a environment method,
        #  or make environment randomization methods take in the scenario,
        #  or just make scenarios more composable and have a few different (static) combinations that are hard-coded
        self.pos3d.enable(Position3DEnableRequest(scoped_link_name=self.ROPE_LINK_NAME, enable=False))
        return self.slide_obstacles(env_rng, params)

    def slide_obstacles(self, env_rng, params: Dict):
        # set random positions for all the objects
        for object_name in params['objects']:
            scoped_link_name = gz_scope(object_name, 'link_1')
            get_res = self.pos3d.get(scoped_link_name=scoped_link_name)
            pos_msg = get_res.pos

            pos_msg.x = pos_msg.x + env_rng.uniform(-0.2, 0.2)
            pos_msg.y = pos_msg.y + env_rng.uniform(-0.2, 0.2)
            pos_msg.z = pos_msg.z + env_rng.uniform(-0.2, 0.2)

            self.register_movable_object(scoped_link_name)
            req = Position3DActionRequest(speed_mps=0.1,
                                          scoped_link_name=scoped_link_name,
                                          tolerance_m=0.01,
                                          position=pos_msg)
            self.pos3d.set(req)

        wait_req = Position3DWaitRequest()
        wait_req.timeout_s = 0.1
        for object_name in params['objects']:
            scoped_link_name = gz_scope(object_name, 'link_1')
            wait_req.scoped_link_names.append(scoped_link_name)
        self.pos3d.wait(wait_req)

        for object_name in params['objects']:
            scoped_link_name = gz_scope(object_name, 'link_1')
            self.pos3d.enable(Position3DEnableRequest(scoped_link_name=scoped_link_name, enable=False))

    @staticmethod
    def integrate_dynamics(s_t: Dict, delta_s_t: Dict):
        return {k: s_t[k] + delta_s_t[k] for k in s_t.keys()}

    def index_action_time(self, action, t):
        action_t = {}
        for feature_name in ['gripper_position']:
            if action[feature_name].is_batched:
                if t < action[feature_name].shape[0]:
                    action_t[feature_name] = action[feature_name][t]
                else:
                    action_t[feature_name] = action[feature_name][t - 1]
            else:
                if t < action[feature_name].shape[1]:
                    action_t[feature_name] = action[feature_name][:, t]
                else:
                    action_t[feature_name] = action[feature_name][:, t - 1]
        return action_t

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

    def plot_goal_rviz(self, goal: Dict, goal_threshold: float, actually_at_goal: Optional[bool] = None):
        goal_marker_msg = MarkerArray()
        tail_marker = Marker()
        tail_marker.scale.x = goal_threshold * 2
        tail_marker.scale.y = goal_threshold * 2
        tail_marker.scale.z = goal_threshold * 2
        tail_marker.action = Marker.ADD
        tail_marker.type = Marker.SPHERE
        tail_marker.header.frame_id = "world"
        tail_marker.header.stamp = rospy.Time.now()
        tail_marker.ns = 'goal'
        tail_marker.id = 0
        if actually_at_goal:
            tail_marker.color.r = 0.4
            tail_marker.color.g = 0.8
            tail_marker.color.b = 0.4
            tail_marker.color.a = 0.8
        else:
            tail_marker.color.r = 0.5
            tail_marker.color.g = 0.3
            tail_marker.color.b = 0.8
            tail_marker.color.a = 0.8
        tail_marker.pose.position.x = goal['tail'][0]
        tail_marker.pose.position.y = goal['tail'][1]
        tail_marker.pose.position.z = goal['tail'][2]
        tail_marker.pose.orientation.w = 1

        goal_marker_msg.markers.append(tail_marker)
        self.state_viz_pub.publish(goal_marker_msg)

    def get_environment(self, params: Dict, **kwargs):
        res = params.get("res", 0.01)
        return get_environment_for_extents_3d(extent=params['extent'],
                                              res=res,
                                              service_provider=self.service_provider,
                                              excluded_models=self.get_excluded_models_for_env())

    def get_excluded_models_for_env(self):
        return ['dragging_rope']

    def needs_reset(self, state: Dict, params: Dict):
        return False
