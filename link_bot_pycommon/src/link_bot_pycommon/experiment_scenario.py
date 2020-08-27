from typing import Dict

import numpy as np
import rospy
from geometry_msgs.msg import Vector3
from std_msgs.msg import Int64, Float32

from peter_msgs.srv import GetPosition3DRequest, Position3DEnableRequest, Position3DActionRequest


class ExperimentScenario:
    def __init__(self):
        self.time_viz_pub = rospy.Publisher("rviz_anim/time", Int64, queue_size=10, latch=True)
        self.traj_idx_viz_pub = rospy.Publisher("traj_idx_viz", Float32, queue_size=10, latch=True)

    def __eq__(self, other):
        if isinstance(other, str):
            return other == self.simple_name()
        raise NotImplementedError()

    @staticmethod
    def simple_name():
        raise NotImplementedError()

    def execute_action(self, action: Dict):
        raise NotImplementedError()

    def sample_action(self,
                      action_rng: np.random.RandomState,
                      environment: Dict,
                      state: Dict,
                      data_collection_params: Dict,
                      action_params: Dict):
        raise NotImplementedError()

    def sample_actions(self,
                       environment: Dict,
                       start_state: Dict,
                       params: Dict,
                       n_action_sequences: int,
                       action_sequence_length: int,
                       rng: np.random.RandomState):
        action_sequences = []

        for _ in range(n_action_sequences):
            action_sequence = []
            for __ in range(action_sequence_length):
                action = self.sample_action(action_rng=rng,
                                            environment=environment,
                                            state=start_state,
                                            data_collection_params=params,
                                            action_params=params)
                action_sequence.append(action)
            action_sequences.append(action_sequence)
        return action_sequences

    @staticmethod
    def local_environment_center_differentiable(state):
        raise NotImplementedError()

    @staticmethod
    def plot_goal_rviz(ax, goal, color, label=None, **kwargs):
        raise NotImplementedError()

    def plot_environment_rviz(self, data: Dict):
        raise NotImplementedError()

    def plot_state_rviz(self, data: Dict, label: str, **kwargs):
        raise NotImplementedError()

    def plot_action_rviz(self, state: Dict, action: Dict, **kwargs):
        raise NotImplementedError()

    def plot_is_close(self, label_t):
        raise NotImplementedError()

    def animate_rviz(self, environment, actual_states, predicted_states, actions, labels, accept_probabilities):
        raise NotImplementedError()

    @staticmethod
    def state_to_points_for_cc(state: Dict):
        raise NotImplementedError()

    def sample_goal(self, environment: Dict, rng: np.random.RandomState, planner_params: Dict):
        raise NotImplementedError()

    @staticmethod
    def distance_to_goal(state, goal):
        raise NotImplementedError()

    @staticmethod
    def distance(s1, s2):
        raise NotImplementedError()

    def __repr__(self):
        raise NotImplementedError()

    @staticmethod
    def robot_name():
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
    def random_object_position(w: float, h: float, c: float, padding: float, rng: np.random.RandomState) -> Dict:
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

    def move_objects_randomly(self, env_rng, movable_objects_services, movable_objects, kinematic: bool, timeout: float = 0.5):
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
        move_action_req.timeout = timeout
        movable_object_services['move'](move_action_req)

    def states_description(self) -> Dict:
        raise NotImplementedError()

    @staticmethod
    def actions_description() -> Dict:
        raise NotImplementedError()

    @staticmethod
    def put_action_local_frame(state: Dict, action: Dict):
        raise NotImplementedError()

    def get_state(self):
        raise NotImplementedError()

    @staticmethod
    def index_state_time(state: Dict, t: int):
        raise NotImplementedError()

    @staticmethod
    def index_predicted_state_time(state: Dict, t: int):
        raise NotImplementedError()

    @staticmethod
    def index_action_time(action: Dict, t: int):
        raise NotImplementedError()

    @staticmethod
    def index_label_time(example: Dict, t: int):
        raise NotImplementedError()

    def randomization_initialization(self):
        raise NotImplementedError()

    def randomize_environment(self, env_rng: np.random.RandomState, objects_params: Dict, data_collection_params: Dict):
        raise NotImplementedError()

    def plot_traj_idx_rviz(self, traj_idx):
        msg = Float32()
        msg.data = traj_idx
        self.traj_idx_viz_pub.publish(msg)

    def plot_time_idx_rviz(self, time_idx):
        msg = Int64()
        msg.data = time_idx
        self.time_viz_pub.publish(msg)

    def dynamics_dataset_metadata(self):
        return {}


def sample_object_position(env_rng, xyz_range: Dict) -> Dict:
    x_range = xyz_range['x']
    y_range = xyz_range['y']
    z_range = xyz_range['z']
    position = Vector3()
    position.x = env_rng.uniform(*x_range)
    position.y = env_rng.uniform(*y_range)
    position.z = env_rng.uniform(*z_range)
    return position


def sample_object_positions(env_rng, movable_objects: Dict) -> Dict[str, Dict]:
    random_object_positions = {name: sample_object_position(
        env_rng, xyz_range) for name, xyz_range in movable_objects.items()}
    return random_object_positions
