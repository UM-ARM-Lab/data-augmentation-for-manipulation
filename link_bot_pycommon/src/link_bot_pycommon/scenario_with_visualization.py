from abc import ABC
from typing import Dict, List

import numpy as np
from matplotlib import cm, colors
from more_itertools import interleave

import ros_numpy
import rospy
from arm_gazebo_msgs.srv import SetModelStatesRequest, SetModelStates
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, GetModelStateRequest, GetModelStateResponse
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Pose, Point, Quaternion
from jsk_recognition_msgs.msg import BoundingBox
from link_bot_data.dataset_utils import NULL_PAD_VALUE
from link_bot_data.visualization import make_delete_marker
from link_bot_pycommon import grid_utils
from link_bot_pycommon.bbox_visualization import extent_to_bbox
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.grid_utils import environment_to_vg_msg, occupied_voxels_to_points
from link_bot_pycommon.marker_index_generator import marker_index_generator
from link_bot_pycommon.rviz_marker_manager import RVizMarkerManager
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from peter_msgs.msg import LabelStatus
from peter_msgs.srv import WorldControl, WorldControlRequest
from rviz_voxelgrid_visuals_msgs.msg import VoxelgridStamped
from std_msgs.msg import Float32, ColorRGBA
from tf import transformations
from visualization_msgs.msg import MarkerArray, Marker


class ScenarioWithVisualization(ExperimentScenario, ABC):
    """
    A lot of our visualization code takes ExperimentScenario as an argument, or at least uses it as a type hint.
    Therefore, in order to avoid circular dependency between the base ExperimentScenario class and visualization code,
    we introduce this class. This class can safely depend on all sorts of visualization code
    """

    def __init__(self):
        super().__init__()
        self.world_control_srv = rospy.ServiceProxy("gz_world_control", WorldControl)
        self.env_viz_pub = rospy.Publisher('occupancy', VoxelgridStamped, queue_size=10)
        self.env_bbox_pub = rospy.Publisher('env_bbox', BoundingBox, queue_size=10)
        self.obs_bbox_pub = rospy.Publisher('obs_bbox', BoundingBox, queue_size=10)
        self.label_viz_pub = rospy.Publisher("label_viz", LabelStatus, queue_size=10)
        self.error_pub = rospy.Publisher("error", Float32, queue_size=10)
        self.point_pub = rospy.Publisher("point", Marker, queue_size=10)

        self.sampled_goal_marker_idx = 0
        self.tree_state_idx = 0
        self.rejected_state_idx = 0
        self.maybe_rejected_state_idx = 0
        self.current_tree_state_idx = 0
        self.tree_action_idx = 0
        self.sample_idx = 0

        self.set_model_states_srv = rospy.ServiceProxy("arm_gazebo/set_model_states", SetModelStates)
        self.set_model_state_srv = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        self.get_model_state_srv = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)

        self.mm = RVizMarkerManager()

    def settle(self):
        req = WorldControlRequest()
        req.seconds = 5
        self.world_control_srv(req)

    @staticmethod
    def random_pos(action_rng: np.random.RandomState, extent):
        x_min, x_max, y_min, y_max, z_min, z_max = extent
        pos = action_rng.uniform([x_min, y_min, z_min], [x_max, y_max, z_max])
        return pos

    def reset_planning_viz(self):
        clear_msg = MarkerArray()
        clear_marker_msg = Marker()
        clear_marker_msg.action = Marker.DELETEALL
        clear_msg.markers.append(clear_marker_msg)
        self.state_viz_pub.publish(clear_msg)
        self.action_viz_pub.publish(clear_msg)
        self.sampled_goal_marker_idx = 0
        self.tree_state_idx = 0
        self.rejected_state_idx = 0
        self.maybe_rejected_state_idx = 0
        self.current_tree_state_idx = 0
        self.tree_action_idx = 0
        self.sample_idx = 0

    def plot_environment_rviz(self, environment: Dict, **kwargs):
        frame = 'env_vg'

        env_msg = environment_to_vg_msg(environment, frame=frame)
        self.env_viz_pub.publish(env_msg)
        vg_points = occupied_voxels_to_points(environment['env'], environment['res'], environment['origin_point'])

        # self.send_occupancy_tf(environment, frame)
        # self.tf.send_transform(environment['origin_point'], [0, 0, 0, 1], 'world', child='origin_point')
        # self.plot_points_rviz(vg_points, label="debugging_vg", frame_id='world', scale=0.002, color='white')

        bbox_msg = extent_to_bbox(environment['extent'])
        bbox_msg.header.frame_id = frame
        self.env_bbox_pub.publish(bbox_msg)

    def send_occupancy_tf(self, environment: Dict, frame):
        grid_utils.send_voxelgrid_tf_origin_point_res(self.tf.tf_broadcaster,
                                                      environment['origin_point'],
                                                      environment['res'],
                                                      frame=frame)

    def plot_executed_action(self, state: Dict, action: Dict, **kwargs):
        self.plot_action_rviz(state, action, label='executed action', color="#3876EB", idx1=1, idx2=1, **kwargs)

    def plot_sampled_goal_state(self, state: Dict):
        self.plot_state_rviz(state, idx=self.sampled_goal_marker_idx, label="goal_sample", color='#EB322F')
        self.sampled_goal_marker_idx += 1

    def plot_start_state(self, state: Dict):
        self.plot_state_rviz(state, label='start', color='#0088aa')

    def plot_sampled_state(self, state: Dict):
        self.plot_state_rviz(state, idx=self.sample_idx, label='samples', color='#f52f32')
        self.sample_idx += 1

    def plot_rejected_state(self, state: Dict):
        self.plot_state_rviz(state, idx=self.rejected_state_idx, label='rejected', color='#ff8822')
        self.rejected_state_idx += 1

    def plot_maybe_rejected_state(self, state: Dict):
        self.plot_state_rviz(state, idx=self.maybe_rejected_state_idx, label='rejected', color='#fac57f')
        self.maybe_rejected_state_idx += 1

    def plot_current_tree_state(self, state: Dict, **kwargs):
        self.plot_state_rviz(state, idx=1, label='current_tree_state', **kwargs)

    def plot_current_tree_action(self, state: Dict, action: Dict, **kwargs):
        self.plot_action_rviz(state, action, idx=1, label='current_tree_action', **kwargs)

    def plot_tree_state(self, state: Dict, **kwargs):
        self.plot_state_rviz(state, idx=self.tree_state_idx, label='tree', **kwargs)
        self.tree_state_idx += 1

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

    def plot_state_closest_to_goal(self, state: Dict, color='#00C282'):
        self.plot_state_rviz(state, label='best', color=color)

    def plot_is_close(self, label_t):
        msg = LabelStatus()
        if label_t is None:
            msg.status = LabelStatus.NA
        elif label_t:
            msg.status = LabelStatus.Accept
        else:
            msg.status = LabelStatus.Reject
        self.label_viz_pub.publish(msg)

    def animate_evaluation_results(self,
                                   environment: Dict,
                                   actual_states: List[Dict],
                                   predicted_states: List[Dict],
                                   actions: List[Dict],
                                   goal: Dict,
                                   goal_threshold: float,
                                   labeling_params: Dict,
                                   accept_probabilities,
                                   horizon: int):
        time_steps = np.arange(len(actual_states))
        self.plot_environment_rviz(environment)
        self.plot_goal_rviz(goal, goal_threshold)

        anim = RvizAnimationController(time_steps)

        while not anim.done:
            t = anim.t()
            s_t = actual_states[t]
            s_t_pred = predicted_states[t]
            self.plot_state_rviz(s_t, label='actual', color='#ff0000aa')
            if horizon is None or 'num_diverged' not in s_t_pred:
                c = '#0000ffaa'
            else:
                c = cm.Blues(s_t_pred['num_diverged'][0] / horizon)
            self.plot_state_rviz(s_t_pred, label='predicted', color=c)
            if len(actions) > 0:
                if t < anim.max_t:
                    self.plot_action_rviz(s_t, actions[t])
                else:
                    self.plot_action_rviz(actual_states[t - 1], actions[t - 1])

            is_close = self.compute_label(s_t, s_t_pred, labeling_params)
            self.plot_is_close(is_close)

            actually_at_goal = self.distance_to_goal(s_t, goal) < goal_threshold
            self.plot_goal_rviz(goal, goal_threshold, actually_at_goal)

            if accept_probabilities and t > 0:
                self.plot_accept_probability(accept_probabilities[t - 1])
            else:
                self.plot_accept_probability(NULL_PAD_VALUE)

            anim.step()

    def animate_rviz(self,
                     environment: Dict,
                     actual_states: List[Dict],
                     predicted_states: List[Dict],
                     actions: List[Dict],
                     labels,
                     accept_probabilities):
        time_steps = np.arange(len(actual_states))
        self.plot_environment_rviz(environment)

        anim = RvizAnimationController(time_steps)

        while not anim.done:
            t = anim.t()
            # FIXME: this assumes lists of states and actions, but in most places we have dicts?
            #  we might be able to deduplicate this code
            s_t = actual_states[t]
            s_t_pred = predicted_states[t]
            self.plot_state_rviz(s_t, label='actual', color='#ff0000aa')
            self.plot_state_rviz(s_t_pred, label='predicted', color='#0000ffaa')
            if t < anim.max_t:
                self.plot_action_rviz(s_t, actions[t])
            else:
                self.plot_action_rviz(actual_states[t - 1], actions[t - 1])

            if labels is not None:
                self.plot_is_close(labels[t])

            if accept_probabilities and t > 0:
                self.plot_accept_probability(accept_probabilities[t - 1])
            else:
                self.plot_accept_probability(NULL_PAD_VALUE)

            anim.step()

    @staticmethod
    def get_environment_from_state_dict(start_states: Dict):
        return {}

    @staticmethod
    def get_environment_from_example(example: Dict):
        if isinstance(example, tuple):
            example = example[0]

        return {
            'env':    example['env'],
            'origin': example['origin'],
            'res':    example['res'],
            'extent': example['extent'],
        }

    def jitter_object_poses(self, env_rng: np.random.RandomState, params: Dict):
        poses = params['environment_randomization']['nominal_poses']

        def _to_pose_msg(p):
            position = ros_numpy.msgify(Point, np.array(p['position']))
            q = np.array(transformations.quaternion_from_euler(*p['orientation']))
            orientation = ros_numpy.msgify(Quaternion, q)
            return Pose(position=position, orientation=orientation)

        poses = {k: _to_pose_msg(v) for k, v in poses.items()}
        random_object_poses = {k: self.jitter_object_pose(pose, env_rng, params) for k, pose in poses.items()}
        return random_object_poses

    def random_new_object_poses(self, env_rng: np.random.RandomState, params: Dict):
        objects = params['environment_randomization']['objects']
        random_object_poses = {k: self.random_object_pose(env_rng, params) for k in objects}
        return random_object_poses

    def jitter_object_pose(self, nominal_object_pose: Pose, env_rng: np.random.RandomState, objects_params: Dict):
        extent = objects_params['environment_randomization']['jitter_extent']
        jitter_pose = self.random_pose_in_extents(env_rng, extent)
        nominal_object_pose.position.x += jitter_pose.position.x
        nominal_object_pose.position.y += jitter_pose.position.y
        nominal_object_pose.position.z += jitter_pose.position.z
        return nominal_object_pose

    def random_object_pose(self, env_rng: np.random.RandomState, objects_params: Dict):
        extent = objects_params['environment_randomization']['extent']
        bbox_msg = extent_to_bbox(extent)
        bbox_msg.header.frame_id = 'world'
        self.obs_bbox_pub.publish(bbox_msg)

        return self.random_pose_in_extents(env_rng, extent)

    def random_pose_in_extents(self, env_rng: np.random.RandomState, extent):
        extent = np.array(extent).reshape(3, 2)
        pose = Pose()
        pose.position = ros_numpy.msgify(Point, env_rng.uniform(extent[:, 0], extent[:, 1]))
        # TODO: make angles configurable
        yaw = env_rng.uniform(-np.pi, np.pi)
        pose.orientation = ros_numpy.msgify(Quaternion, transformations.quaternion_from_euler(0, 0, yaw))
        return pose

    def set_object_poses(self, object_positions: Dict):
        set_states_req = SetModelStatesRequest()
        for object_name, pose in object_positions.items():
            state = ModelState()
            state.model_name = object_name
            state.pose = pose
            set_states_req.model_states.append(state)
        self.set_model_states_srv(set_states_req)

    def get_object_poses(self, names: List):
        poses = {}
        for object_name in names:
            get_req = GetModelStateRequest()
            get_req.model_name = object_name
            res: GetModelStateResponse = self.get_model_state_srv(get_req)
            poses[object_name] = res.pose
        return poses

    def animate_final_path(self,
                           environment: Dict,
                           planned_path: List[Dict],
                           actions: List[Dict]):
        time_steps = np.arange(len(planned_path))
        self.plot_environment_rviz(environment)

        anim = RvizAnimationController(time_steps)

        while not anim.done:
            t = anim.t()
            s_t_planned = planned_path[t]
            self.plot_state_rviz(s_t_planned, label='planned', color='#FF4616')
            if len(actions) > 0:
                if t < anim.max_t:
                    self.plot_action_rviz(s_t_planned, actions[t])
                else:
                    self.plot_action_rviz(planned_path[t - 1], actions[t - 1])

            anim.step()

    def plot_point_rviz(self, position, label: str, frame_id: str = 'world', id: int = 0, scale: float = 0.02):
        msg = Marker()
        msg.header.frame_id = frame_id
        msg.header.stamp = rospy.Time.now()
        msg.ns = label
        msg.id = id
        msg.pose.position.x = position[0]
        msg.pose.position.y = position[1]
        msg.pose.position.z = position[2]
        msg.pose.orientation.w = 1
        msg.type = Marker.SPHERE
        msg.action = Marker.ADD
        msg.scale.x = scale
        msg.scale.y = scale
        msg.scale.z = scale
        msg.color.r = 1
        msg.color.g = 1
        msg.color.b = 0
        msg.color.a = 1

        self.point_pub.publish(msg)

    def plot_points_rviz(self, positions, label: str, frame_id: str = 'world', id: int = 0, **kwargs):
        color_msg = ColorRGBA(*colors.to_rgba(kwargs.get("color", "r")))

        scale = kwargs.get('scale', 0.02)

        msg = Marker()
        msg.header.frame_id = frame_id
        msg.header.stamp = rospy.Time.now()
        msg.ns = label
        msg.id = id
        msg.type = Marker.SPHERE_LIST
        msg.action = Marker.ADD
        msg.pose.orientation.w = 1
        msg.scale.x = scale
        msg.scale.y = scale
        msg.scale.z = scale
        msg.color = color_msg
        for position in positions:
            p = Point(x=position[0], y=position[1], z=position[2])
            msg.points.append(p)

        self.point_pub.publish(msg)

    def plot_lines_rviz(self, starts, ends, label: str, frame_id: str = 'world', id: int = 0, **kwargs):
        if starts is None or ends is None:
            return

        color_msg = ColorRGBA(*colors.to_rgba(kwargs.get("color", "y")))

        scale = kwargs.get('scale', 0.001)

        msg = Marker()
        msg.header.frame_id = frame_id
        msg.header.stamp = rospy.Time.now()
        msg.ns = label
        msg.id = id
        msg.type = Marker.LINE_LIST
        msg.action = Marker.ADD
        msg.pose.orientation.w = 1
        msg.scale.x = scale
        msg.scale.y = scale
        msg.scale.z = scale
        msg.color = color_msg
        for position in interleave(starts, ends):
            p = Point(x=position[0], y=position[1], z=position[2])
            msg.points.append(p)

        self.point_pub.publish(msg)

    def delete_points_rviz(self, label: str, id: int = 0):
        self.point_pub.publish(make_delete_marker(id, label))

    def delete_lines_rviz(self, label: str, id: int = 0):
        self.point_pub.publish(make_delete_marker(id, label))

    def plot_error_rviz(self, error):
        self.error_pub.publish(Float32(data=error))
