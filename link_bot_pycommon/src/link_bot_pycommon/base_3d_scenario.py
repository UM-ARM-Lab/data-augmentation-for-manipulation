from typing import Dict, List

import numpy as np
from matplotlib import cm

import rospy
from gazebo_msgs.srv import GetModelState, GetModelStateRequest
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
from link_bot_data.dataset_utils import NULL_PAD_VALUE
from link_bot_pycommon import grid_utils
from link_bot_pycommon.bbox_visualization import extent_to_bbox
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.grid_utils import environment_to_occupancy_msg
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from mps_shape_completion_msgs.msg import OccupancyStamped
from peter_msgs.msg import LabelStatus
from peter_msgs.srv import WorldControl, WorldControlRequest
from tf import transformations
from visualization_msgs.msg import MarkerArray, Marker

try:
    from jsk_recognition_msgs.msg import BoundingBox
except ImportError:
    rospy.logwarn("ignoring failed import of BBox message")


class Base3DScenario(ExperimentScenario):
    def __init__(self):
        super().__init__()
        self.world_control_srv = rospy.ServiceProxy("/world_control", WorldControl)
        self.env_viz_pub = rospy.Publisher('occupancy', OccupancyStamped, queue_size=10, latch=True)
        try:
            self.env_bbox_pub = rospy.Publisher('env_bbox', BoundingBox, queue_size=10, latch=True)
            self.obs_bbox_pub = rospy.Publisher('obs_bbox', BoundingBox, queue_size=10, latch=True)
        except NameError:
            pass
        self.state_viz_pub = rospy.Publisher("state_viz", MarkerArray, queue_size=10, latch=True)
        self.action_viz_pub = rospy.Publisher("action_viz", MarkerArray, queue_size=10, latch=True)
        self.label_viz_pub = rospy.Publisher("label_viz", LabelStatus, queue_size=10, latch=True)

        self.sampled_goal_marker_idx = 0
        self.tree_state_idx = 0
        self.rejected_state_idx = 0
        self.maybe_rejected_state_idx = 0
        self.current_tree_state_idx = 0
        self.tree_action_idx = 0
        self.sample_idx = 0

        self.set_model_state_srv = rospy.ServiceProxy("gazebo/set_model_state", SetModelState)
        self.get_model_state_srv = rospy.ServiceProxy("gazebo/get_model_state", GetModelState)

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
        from time import sleep
        for i in range(10):
            self.state_viz_pub.publish(clear_msg)
            self.action_viz_pub.publish(clear_msg)
            sleep(0.1)
        self.sampled_goal_marker_idx = 0
        self.tree_state_idx = 0
        self.rejected_state_idx = 0
        self.maybe_rejected_state_idx = 0
        self.current_tree_state_idx = 0
        self.tree_action_idx = 0
        self.sample_idx = 0

    def plot_environment_rviz(self, data: Dict):
        self.send_occupancy_tf(data)

        env_msg = environment_to_occupancy_msg(data)
        self.env_viz_pub.publish(env_msg)

        bbox_msg = extent_to_bbox(data['extent'])
        bbox_msg.header.frame_id = 'world'
        self.env_bbox_pub.publish(bbox_msg)

    def send_occupancy_tf(self, environment: Dict):
        grid_utils.send_occupancy_tf(self.tf.tf_broadcaster, environment)

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

    def plot_current_tree_state(self, state: Dict, horizon: int, **kwargs):
        self.plot_state_rviz(state, idx=1, label='current_tree_state', **kwargs)

    def plot_tree_state(self, state: Dict, color='#777777'):
        self.plot_state_rviz(state, idx=self.tree_state_idx, label='tree', color=color)
        self.tree_state_idx += 1

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

    def random_new_object_poses(self, env_rng: np.random.RandomState, params: Dict):
        random_object_poses = {k: self.random_object_pose(env_rng, params) for k in params['objects']}
        return random_object_poses

    def random_pose_in_extents(self, env_rng: np.random.RandomState, extent):
        extent = np.array(extent).reshape(3, 2)
        position = env_rng.uniform(extent[:, 0], extent[:, 1])
        yaw = env_rng.uniform(-np.pi, np.pi)
        orientation = transformations.quaternion_from_euler(0, 0, yaw)
        return (position, orientation)

    def random_object_pose(self, env_rng: np.random.RandomState, objects_params: Dict):
        extent = objects_params['objects_extent']
        bbox_msg = extent_to_bbox(extent)
        bbox_msg.header.frame_id = 'world'
        self.obs_bbox_pub.publish(bbox_msg)

        return self.random_pose_in_extents(env_rng, extent)

    def set_object_poses(self, object_positions: Dict):
        for object_name, (position, orientation) in object_positions.items():
            set_req = SetModelStateRequest()
            set_req.model_state.model_name = object_name
            set_req.model_state.pose.position.x = position[0]
            set_req.model_state.pose.position.y = position[1]
            set_req.model_state.pose.position.z = position[2]
            set_req.model_state.pose.orientation.x = orientation[0]
            set_req.model_state.pose.orientation.y = orientation[1]
            set_req.model_state.pose.orientation.z = orientation[2]
            set_req.model_state.pose.orientation.w = orientation[3]
            self.set_model_state_srv(set_req)

    def get_object_poses(self, names: List):
        poses = {}
        for object_name in names:
            get_req = GetModelStateRequest()
            get_req.model_name = object_name
            res = self.get_model_state_srv(get_req)
            poses[object_name] = res
        return poses