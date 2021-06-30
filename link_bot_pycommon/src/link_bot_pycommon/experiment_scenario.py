import logging
from typing import Dict, Optional

import numpy as np

import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from geometry_msgs.msg import Vector3
from link_bot_data.dataset_utils import add_predicted
from link_bot_pycommon.base_services import BaseServices
from link_bot_pycommon.heartbeat import HeartBeat
from link_bot_pycommon.sample_object_positions import sample_object_position, sample_object_positions
from moonshine.indexing import index_dict_of_batched_tensors_tf, index_time_2
from peter_msgs.srv import GetPosition3DRequest, Position3DEnableRequest, Position3DActionRequest
from std_msgs.msg import Int64, Float32
from visualization_msgs.msg import MarkerArray

logger = logging.getLogger(__file__)


class MockRobot:
    def __init__(self):
        self.robot_namespace = 'mock_robot'


class ExperimentScenario:
    def __init__(self):
        self.tf_features_converters = {}
        self.time_viz_pub = rospy.Publisher("rviz_anim/time", Int64, queue_size=10, latch=True)
        self.traj_idx_viz_pub = rospy.Publisher("traj_idx_viz", Float32, queue_size=10, latch=True)
        self.recovery_prob_viz_pub = rospy.Publisher("recovery_probability_viz", Float32, queue_size=10, latch=True)
        self.accept_probability_viz_pub = rospy.Publisher("accept_probability_viz", Float32, queue_size=10, latch=True)
        self.stdev_viz_pub = rospy.Publisher("stdev", Float32, queue_size=10)
        self.state_viz_pub = rospy.Publisher("state_viz", MarkerArray, queue_size=10, latch=True)
        self.action_viz_pub = rospy.Publisher("action_viz", MarkerArray, queue_size=10, latch=True)
        self.h = HeartBeat()

        self.tf = TF2Wrapper()
        self.robot = MockRobot()

    @staticmethod
    def simple_name():
        raise NotImplementedError()

    def execute_action(self, environment, state, action: Dict) -> bool:
        raise NotImplementedError()

    @staticmethod
    def add_action_noise(action: Dict, noise_rng: np.random.RandomState):
        return action

    def is_action_valid(self, environment: Dict, state: Dict, action: Dict, action_params: Dict):
        return True

    def sample_action(self,
                      action_rng: np.random.RandomState,
                      environment: Dict,
                      state: Dict,
                      action_params: Dict,
                      validate, stateless: Optional[bool] = False):
        raise NotImplementedError()

    def sample_action_sequences(self,
                                environment: Dict,
                                state: Dict,
                                action_params: Dict,
                                n_action_sequences: int,
                                action_sequence_length: int,
                                validate: bool,
                                action_rng: np.random.RandomState):
        action_sequences = []

        for _ in range(n_action_sequences):
            action_sequence = self.sample_action_batch(environment=environment,
                                                       state=state,
                                                       action_params=action_params,
                                                       batch_size=action_sequence_length,
                                                       validate=validate,
                                                       action_rng=action_rng)
            action_sequences.append(action_sequence)
        return action_sequences

    def sample_action_batch(self,
                            environment: Dict,
                            state: Dict,
                            action_params: Dict,
                            batch_size: int,
                            validate: bool,
                            action_rng: np.random.RandomState):
        action_sequence = []
        for i in range(batch_size):
            action, invalid = self.sample_action(action_rng=action_rng,
                                                 environment=environment,
                                                 state=state,
                                                 action_params=action_params,
                                                 validate=validate,
                                                 stateless=True)
            action_sequence.append(action)
        return action_sequence

    @staticmethod
    def interpolate(start_state, end_state, step_size=0.05):
        raise NotImplementedError()

    @staticmethod
    def local_environment_center_differentiable(state):
        raise NotImplementedError()

    def plot_goal_rviz(self, goal: Dict, goal_threshold: float, actually_at_goal: Optional[bool] = None):
        raise NotImplementedError()

    def plot_environment_rviz(self, data: Dict, **kwargs):
        raise NotImplementedError()

    def delete_state_rviz(self, label: str, index: int):
        raise NotImplementedError()

    def plot_state_rviz(self, data: Dict, **kwargs):
        raise NotImplementedError()

    def plot_action_rviz(self, state: Dict, action: Dict, **kwargs):
        raise NotImplementedError()

    def delete_action_rviz(self, label: str, index: int):
        raise NotImplementedError()

    def plot_is_close(self, label_t):
        raise NotImplementedError()

    def plot_traj_idx_rviz(self, traj_idx):
        msg = Float32()
        msg.data = traj_idx
        self.traj_idx_viz_pub.publish(msg)

    def plot_time_idx_rviz(self, time_idx):
        msg = Int64()
        msg.data = time_idx
        self.time_viz_pub.publish(msg)

    def plot_recovery_probability(self, recovery_probability: float):
        msg = Float32()
        msg.data = recovery_probability
        self.recovery_prob_viz_pub.publish(msg)

    def plot_recovery_probability_t(self, example: Dict, t: int):
        self.plot_recovery_probability(example['recovery_probability'][t])

    def plot_accept_probability_t(self, example: Dict, t: int):
        accept_probability_t = index_time_2(example, 'accept_probability', t)
        self.plot_accept_probability(accept_probability_t)

    def plot_accept_probability(self, accept_probability_t: float):
        msg = Float32()
        msg.data = accept_probability_t
        self.accept_probability_viz_pub.publish(msg)

    def plot_dynamics_stdev_t(self, example: Dict, t: int):
        stdev_t = example[add_predicted('stdev')][t]
        self.plot_stdev(stdev_t)

    def plot_stdev(self, stdev_t: float):
        msg = Float32()
        msg.data = stdev_t
        self.stdev_viz_pub.publish(msg)

    def animate_rviz(self, environment, actual_states, predicted_states, actions, labels, accept_probabilities):
        raise NotImplementedError()

    @staticmethod
    def state_to_points_for_cc(state: Dict):
        raise NotImplementedError()

    def sample_goal(self, environment: Dict, rng: np.random.RandomState, planner_params: Dict):
        raise NotImplementedError()

    def distance_to_goal(self, state: Dict, goal: Dict):
        raise NotImplementedError()

    def distance_to_goal_state(self, state: Dict, goal_state: Dict, goal_type: str):
        goal = self.goal_state_to_goal(goal_state, goal_type)
        return self.distance_to_goal(state, goal)

    def goal_state_to_goal(self, goal_state: Dict, goal_type: str) -> Dict:
        raise NotImplementedError()

    def classifier_distance(self, s1: Dict, s2: Dict):
        """ this is not the distance metric used in planning """
        raise NotImplementedError()

    def compute_label(self, actual: Dict, predicted: Dict, labeling_params: Dict):
        model_error = self.classifier_distance(actual, predicted)
        threshold = labeling_params['threshold']
        is_close = model_error < threshold
        return is_close

    def __repr__(self):
        raise NotImplementedError()

    @staticmethod
    def dynamics_loss_function(dataset_element, predictions):
        raise NotImplementedError()

    @staticmethod
    def dynamics_metrics_function(dataset_element, predictions):
        raise NotImplementedError()

    @staticmethod
    def integrate_dynamics(s_t, ds_t):
        raise NotImplementedError()

    @staticmethod
    def get_environment_from_example(example: Dict):
        raise NotImplementedError()

    @staticmethod
    def get_environment_from_state_dict(start_states: Dict):
        raise NotImplementedError()

    @staticmethod
    def put_state_local_frame(state: Dict):
        raise NotImplementedError()

    @staticmethod
    def put_state_robot_frame(state: Dict):
        raise NotImplementedError()

    @staticmethod
    def random_object_position(w: float, h: float, c: float, padding: float, rng: np.random.RandomState):
        xyz_range = {
            'x': [-w / 2 + padding, w / 2 - padding],
            'y': [-h / 2 + padding, h / 2 - padding],
            'z': [-c / 2 + padding, c / 2 - padding],
        }
        return sample_object_position(rng, xyz_range)

    @staticmethod
    def get_movable_object_positions(movable_objects_services: Dict):
        positions = {}
        for object_name, services in movable_objects_services.items():
            position_response = services['get_position'](GetPosition3DRequest())
            positions[object_name] = position_response
        return positions

    def move_objects_randomly(self, env_rng, movable_objects_services, movable_objects, kinematic: bool,
                              timeout: float = 0.5):
        random_object_positions = sample_object_positions(env_rng, movable_objects)
        if kinematic:
            raise NotImplementedError()
        else:
            ExperimentScenario.move_objects(movable_objects_services, random_object_positions, timeout)

    @staticmethod
    def move_objects_to_positions(movable_objects_services: Dict, object_positions: Dict, timeout: float = 0.5):
        object_positions = {}
        for name, (x, y) in object_positions.items():
            position = Vector3()
            position.x = x
            position.y = y
            object_positions[name] = position
        return ExperimentScenario.move_objects(movable_objects_services, object_positions, timeout)

    @staticmethod
    def set_objects(movable_objects_services: Dict, object_positions: Dict, timeout: float):
        for name, position in object_positions.items():
            services = movable_objects_services[name]
            ExperimentScenario.call_set(services, name, position)

    @staticmethod
    def move_objects(movable_objects_services: Dict, object_positions: Dict, timeout: float):
        for name, position in object_positions.items():
            services = movable_objects_services[name]
            ExperimentScenario.call_move(services, name, position, timeout)

        # disable controller so objects can move around
        for object_name, _ in object_positions.items():
            movable_object_services = movable_objects_services[object_name]
            enable_req = Position3DEnableRequest()
            enable_req.enable = False
            movable_object_services['enable'](enable_req)

    @staticmethod
    def call_set(movable_object_services, object_name, position):
        set_action_req = Position3DActionRequest()
        set_action_req.position = position
        movable_object_services['set'](set_action_req)

    @staticmethod
    def call_move(movable_object_services, object_name, position, timeout):
        move_action_req = Position3DActionRequest()
        move_action_req.position = position
        move_action_req.timeout_s = timeout
        movable_object_services['move'](move_action_req)

    @staticmethod
    def put_action_local_frame(state: Dict, action: Dict):
        raise NotImplementedError()

    def get_state(self):
        raise NotImplementedError()

    def randomization_initialization(self, params: Dict):
        pass

    def randomize_environment(self, env_rng: np.random.RandomState, params: Dict):
        raise NotImplementedError()

    def dynamics_dataset_metadata(self):
        return {}

    def get_environment(self, params: Dict, **kwargs):
        raise NotImplementedError()

    def on_before_action(self):
        pass

    def on_before_get_state_or_execute_action(self):
        pass

    def on_before_data_collection(self, params: Dict):
        raise NotImplementedError()

    def get_excluded_models_for_env(self):
        raise NotImplementedError()

    def cfm_distance(self, z1, z2):
        raise NotImplementedError()

    def on_after_data_collection(self, params):
        pass

    def needs_reset(self, state: Dict, params: Dict):
        raise NotImplementedError()

    def restore_from_bag(self, service_provider: BaseServices, params: Dict, bagfile_name):
        service_provider.restore_from_bag(bagfile_name)

    def heartbeat(self):
        self.h.update()

    def move_objects_out_of_scene(self, params: Dict):
        pass
