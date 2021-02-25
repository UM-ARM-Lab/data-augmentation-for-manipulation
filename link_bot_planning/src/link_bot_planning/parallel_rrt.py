from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Callable, List

import numpy as np
import tensorflow as tf
from matplotlib import cm
from numpy import pi
from tensorflow import random as rng

import rospy
from jsk_recognition_msgs.msg import BoundingBox
from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_planning.my_planner import PlanningQuery, MyPlanner, PlanningResult, MyPlannerStatus
from link_bot_planning.timeout_or_not_progressing import TimeoutOrNotProgressing
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from link_bot_pycommon.bbox_visualization import extent_to_bbox
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.pycommon import make_dict_tf_float32
from moonshine.moonshine_utils import add_batch, repeat, sequence_of_dicts_to_dict_of_tensors, numpify
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction
from state_space_dynamics.base_filter_function import BaseFilterFunction
from tf import transformations


@dataclass
class Node:
    state: Dict
    action: Dict
    parent_index: tf.Tensor
    children_index: tf.Tensor
    children: tf.Tensor


@dataclass
class Nodes:
    states: Dict
    actions: Dict
    parent_indices: tf.Tensor
    children_indices: tf.Tensor
    children: tf.Tensor


def concat_dicts(d1: Dict, d2: Dict, axis: int = 0):
    out_d = {}
    for k in d1.keys():
        out_d[k] = tf.concat([d1[k], d2[k]], axis=axis)
    return out_d


class MyTree:
    MAX_NUM_CHILDREN = 10
    NO_PARENT = -1
    EMPTY_CHILDREN = np.zeros(MAX_NUM_CHILDREN, dtype=np.int64)

    def __init__(self,
                 state_space: DualArmRopeStateSpace,
                 action_space: DualArmRopeActionSpace,
                 state: Dict,
                 batch_size: int):
        self.state_space = state_space
        self.action_space = action_space
        self.batch_size = batch_size

        self.rng = np.random.RandomState(0)

        # TODO: use the dataclasses here?
        self.states = add_batch(state)
        self.actions = add_batch(self.action_space.null_action())
        # the are numpy not tf because numpy is easier to work with, and they don't need to be tf tensors
        self.parent_indices = np.array([self.NO_PARENT], dtype=np.int64)
        self.children = [[]]

    def add(self, parent_indices: tf.Tensor, action: Dict, state: Dict):
        k = parent_indices.shape[0]
        before_size = self.size
        self.states = concat_dicts(self.states, state)
        self.actions = concat_dicts(self.actions, action)
        self.parent_indices = np.concatenate([self.parent_indices, parent_indices], axis=0)
        for _ in range(k):  # for each new node, add an empty list for the children
            self.children.append([])
        after_size = self.size
        new_node_indices = np.arange(before_size, after_size)

        # set the children data for the parents
        for p_i, n_i in zip(parent_indices, new_node_indices):
            self.children[p_i].append(n_i)

        return new_node_indices

    def sample(self):
        selected_indices = self.rng.randint(0, self.size, size=self.batch_size)
        # make this a function?
        state_keys = self.state_space.sizes.keys()
        selected_states = {k: tf.gather(self.states[k], selected_indices, axis=0) for k in state_keys}
        return selected_indices, selected_states

    @property
    def size(self):
        return self.parent_indices.shape[0]


def get_planning_scenario(scenario, planner_params, action_params, state_sampler_rng, action_sampler_rng, plot):
    pass


def tf_nearest(matrix, query, distance_function: Callable):
    distances = distance_function(matrix, tf.expand_dims(query, 0))
    nearest_idx = tf.argmin(distances)
    nearest_x = matrix[nearest_idx]
    return nearest_x


class DualArmRopePlanning:
    def __init__(self,
                 scenario: ExperimentScenario,
                 planner_params: Dict,
                 action_params: Dict,
                 state_sampler_rng: rng.Generator,
                 action_sampler_rng: rng.Generator,
                 plot: bool,
                 ):
        self.s = scenario
        self.planner_params = planner_params
        self.action_params = action_params
        self.state_sampler_rng = state_sampler_rng
        self.action_sampler_rng = action_sampler_rng
        self.plot = plot
        self.state_space = self.make_state_space(self.planner_params, self.state_sampler_rng, self.plot)
        self.action_space = self.make_action_space(self.state_space, self.action_sampler_rng, self.action_params)

    def make_goal_checker(self, rng: rng.Generator, params: Dict, plot: bool):
        raise NotImplementedError()

    def make_state_space(self, planner_params, state_sampler_rng: rng.Generator, plot: bool):
        raise NotImplementedError()

    def make_action_space(self, state_space, rng: rng.Generator, action_params: Dict):
        raise NotImplementedError()


class DualArmRopeGoalChecker:
    def __init__(self, scenario: ExperimentScenario, params: Dict, batch_size: int):
        self.scenario = scenario
        self.batch_size = batch_size
        self.params = params
        self.goal_params = self.params['goal_params']
        self.threshold = self.goal_params['threshold']

    def reached(self, state: Dict, goal: Dict):
        distance = self.distance_to_goal(state, goal)
        return distance < self.threshold

    def distance_to_goal(self, state: Dict, goal: Dict):
        k = state['rope'].shape[0]
        rope_points = tf.reshape(state['rope'], [k, -1, 3])
        # NOTE: well ok not _any_ node, but ones near the middle
        n_from_ends = 7
        point_batched = goal['point'][tf.newaxis, tf.newaxis]
        distances = tf.linalg.norm(point_batched - rope_points, axis=-1)[:, n_from_ends:-n_from_ends]
        distance = tf.reduce_min(distances, axis=-1)
        return distance

    def nearest_to_goal_in_batch(self, states: Dict, goal: Dict):
        dist_to_goal = self.distance_to_goal(states, goal)
        nearest_to_goal_idx = tf.argmin(dist_to_goal)
        nearest_to_goal_dist = dist_to_goal[nearest_to_goal_idx]
        state_nearest_to_goal = {k: v[nearest_to_goal_idx] for k, v in states.items()}
        return state_nearest_to_goal, nearest_to_goal_dist


class DualArmRopeGoalSampler:
    def __init__(self, scenario: ExperimentScenario, params: Dict, rng: rng.Generator):
        self.plot = True
        self.scenario = scenario
        self.params = params
        self.goal_params = self.params['goal_params']
        self.threshold = self.goal_params['threshold']
        self.rng = rng

    def sample(self, goal, batch_size: int):
        random_distance = self.rng.uniform([batch_size], 0.0, self.threshold)
        v = tf.constant([random_distance, 0, 0, 1], tf.float32)
        random_direction = tf.matmul(transformations.random_rotation_matrix(self.rng.uniform(1, 0, 1, [3])), v)
        random_direction = random_direction[:3]
        random_point = goal['point'] + random_direction

        goal_state_sample = self.make_goal_state(random_point)

        if self.plot:
            self.scenario.plot_sampled_goal_state(goal_state_sample)

        return goal_state_sample

    def make_goal_state(self, random_point):
        goal_state = {
            'left_gripper':  random_point,
            'right_gripper': random_point,
            'rope':          [random_point] * self.scenario.n_links,
            'stdev':         np.zeros(1, dtype=tf.float32),
        }
        return goal_state


class DualArmRopeStateSampler:
    def __init__(self, state_space, scenario: ExperimentScenario, params: Dict, rng: rng.Generator, goal_sampler,
                 batch_size):
        self.batch_size = batch_size
        self.plot = True
        self.scenario = scenario
        self.params = params
        self.rng = rng
        self.state_space = state_space

        self.goal_sampler = goal_sampler

        self.extent_flat = self.params['state_sampler_extent']
        self.extent = np.array(self.extent_flat).reshape(3, 2)

        bbox_msg = extent_to_bbox(self.extent_flat)
        bbox_msg.header.frame_id = 'world'
        self.sampler_extents_bbox_pub = rospy.Publisher('sampler_extents', BoundingBox, queue_size=10, latch=True)
        self.sampler_extents_bbox_pub.publish(bbox_msg)

    def sample(self):
        random_point = self.rng.uniform([self.batch_size, 3],
                                        minval=self.extent[:, 0],
                                        maxval=self.extent[:, 1])  # [k, 3]
        random_point_rope = tf.tile(tf.expand_dims(random_point, axis=1), self.scenario.n_links)  # [k, 25, 3]
        random_point_rope = tf.reshape(random_point_rope, [self.batch_size, -1])  # [k, 75]
        sampled_states_dict = {
            'left_gripper':  random_point,
            'right_gripper': random_point,
            'rope':          random_point_rope,
            'stdev':         tf.zeros(1, dtype=tf.float32),
        }
        sampled_states = self.state_space.state_dict_to_vector(sampled_states_dict)

        if self.plot:
            self.scenario.plot_sampled_states(sampled_states)

        return sampled_states


def sample_dual_gripper_action(rng: rng.Generator, action_params: Dict):
    m = action_params['max_distance_gripper_can_move']

    left_phi = rng.uniform(-pi, pi)
    left_theta = rng.uniform(-pi, pi)
    left_r = rng.uniform(0, m)

    right_phi = rng.uniform(-pi, pi)
    right_theta = rng.uniform(-pi, pi)
    right_r = rng.uniform(0, m)

    return [left_r,
            left_phi,
            left_theta,
            right_r,
            right_phi,
            right_theta]


def rpy_to_euler_matrix_tf(roll, pitch, yaw):
    si, sj, sk = tf.math.sin(roll), tf.math.sin(pitch), tf.math.sin(yaw)
    ci, cj, ck = tf.math.cos(roll), tf.math.cos(pitch), tf.math.cos(yaw)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk
    zero = tf.zeros_like(roll)
    one = tf.zeros_like(roll)

    M = tf.stack([tf.stack([cj * ck, sj * sc - cs, sj * cc + ss, zero], axis=1),
                  tf.stack([cj * sk, sj * ss + cc, sj * cs - sc, zero], axis=1),
                  tf.stack([-sj, cj * si, cj * ci, zero], axis=1),
                  tf.stack([zero, zero, zero, one], axis=1)], axis=1)
    return M


def batch_matrix_nonbatch_vector_multiply(A, b):
    """

    Args:
        rotation_matrix: [b, n, m]
        one:  [m]

    Returns:
        [b, n]

    """
    return tf.einsum('bnm,m->bn', A, b)


def sample_delta_position_tf(action_params: Dict, action_rng: rng.Generator, batch_size: int):
    pitch = action_rng.uniform([batch_size], -pi, pi)
    yaw = action_rng.uniform([batch_size], -pi, pi)
    displacement = action_rng.uniform([batch_size, 1], 0, action_params['max_distance_gripper_can_move'])
    roll = tf.zeros_like(pitch)
    # TODO: replace this with tf code
    rotation_matrix = rpy_to_euler_matrix_tf(roll, pitch, yaw)
    one = tf.constant([1, 0, 0, 1], tf.float32)
    gripper_delta_position_homo = batch_matrix_nonbatch_vector_multiply(rotation_matrix, one) * displacement
    gripper_delta_position = gripper_delta_position_homo[:, :3]
    return gripper_delta_position


class DualArmRopeActionSampler:
    def __init__(self, action_space, scenario: ExperimentScenario, params: Dict, action_params: Dict,
                 rng: rng.Generator, batch_size):
        self.action_space = action_space
        self.scenario = scenario
        self.params = params
        self.action_params = action_params
        self.rng = rng
        self.batch_size = batch_size

    def sample(self):
        left_gripper_delta_positions = sample_delta_position_tf(self.action_params, self.rng, self.batch_size)
        right_gripper_delta_positions = sample_delta_position_tf(self.action_params, self.rng, self.batch_size)
        actions = {
            'left_gripper_delta_position':  left_gripper_delta_positions,
            'right_gripper_delta_position': right_gripper_delta_positions,
        }
        return actions


def satisfies_bounds(params: Dict, state: Dict, batch_size: int):
    extent_flat = params['extent']
    extent = tf.cast(tf.reshape(extent_flat, [3, 2]), tf.float32)
    lower = tf.convert_to_tensor(extent[:, 0], dtype=tf.float32)[tf.newaxis, tf.newaxis]
    upper = tf.convert_to_tensor(extent[:, 1], dtype=tf.float32)[tf.newaxis, tf.newaxis]

    left_gripper = tf.cast(tf.reshape(state['left_gripper'], [batch_size, -1, 3]), tf.float32)
    right_gripper = tf.cast(tf.reshape(state['right_gripper'], [batch_size, -1, 3]), tf.float32)
    rope = tf.cast(tf.reshape(state['rope'], [batch_size, -1, 3]), tf.float32)
    all_points = tf.concat([left_gripper, right_gripper, rope], axis=1)
    in_bounds = tf.logical_and(lower < all_points, all_points < upper)
    valid = tf.reduce_all(tf.reduce_all(in_bounds, axis=-1), axis=-1)
    return valid


class DualArmRopeStateSpace:
    def __init__(self, scenario: ExperimentScenario, params: Dict):
        self.scenario = scenario
        self.params = params
        # TODO: de-duplicate this information?
        self.sizes = {
            'left_gripper':    3,
            'right_gripper':   3,
            'rope':            75,
            'joint_positions': 20,
            'joint_names':     20,
            'stdev':           1,
        }
        self.total_dim = sum(self.sizes.values())

    def state_dict_to_vector_batched(self, state: Dict, batch_size: int):
        # NOTE: we force a specific order to prevent silly ordering mistakes
        vectors = []
        for k in self.sizes.keys():
            v = state[k]
            vectors.append(tf.reshape(v, [batch_size, -1]))
        x = tf.concat(vectors, axis=-1)
        return x

    def state_dict_to_vector(self, state: Dict):
        # NOTE: we force a specific order to prevent silly ordering mistakes
        vectors = []
        for k in self.sizes.keys():
            v = state[k]
            vectors.append(tf.reshape(v, [-1]))
        x = tf.concat(vectors, axis=-1)
        return x

    def goal_dict_to_vector(self, goal: Dict):
        return goal['point']

    def vector_to_state_dict_batched(self, x):
        vectors = tf.split(x, list(self.sizes.values()), axis=1)
        state_dict = {k: v for k, v in zip(self.sizes.keys(), vectors)}
        return state_dict

    def vector_to_state_dict(self, x):
        vectors = tf.split(x, list(self.sizes.values()), axis=0)
        state_dict = {k: v for k, v in zip(self.sizes.keys(), vectors)}
        return state_dict


class DualArmRopeActionSpace:
    def __init__(self, scenario: ExperimentScenario, params: Dict):
        self.scenario = scenario
        self.params = params
        self.sizes = {
            'left_gripper_position':  3,
            'right_gripper_position': 3,
        }
        self.total_dim = sum(self.sizes.values())

    def null_action(self):
        return {
            'left_gripper_position':  np.ones([3]) * -10,
            'right_gripper_position': np.ones([3]) * -10,
        }

    def action_dict_to_vector_batched(self, action: Dict, batch_size: int):
        vectors = []
        for k, v in action.items():
            vectors.append(tf.reshape(v, [batch_size, -1]))
        return tf.concat(vectors, axis=0)

    def action_dict_to_vector(self, action: Dict):
        vectors = []
        for k, v in action.items():
            vectors.append(tf.reshape(v, [-1]))
        return tf.concat(vectors, axis=0)

    def vector_to_action_dict_batched(self, x: tf.Tensor):
        vectors = tf.split(x, list(self.sizes.values()), axis=1)
        action_dict = {k: v for k, v in zip(self.sizes.keys(), vectors)}
        return action_dict

    def vector_to_action_dict(self, x: tf.Tensor):
        vectors = tf.split(x, list(self.sizes.values()), axis=0)
        action_dict = {k: v for k, v in zip(self.sizes.keys(), vectors)}
        return action_dict

    def foobar(self, state, action):
        new_action = {
            'left_gripper_position':  state['left_gripper'] + action['left_gripper_delta_position'],
            'right_gripper_position': state['right_gripper'] + action['right_gripper_delta_position'],
        }
        return new_action


def filter_valid_array(valid: tf.Tensor, item: np.array):
    valid_indices = np.where(valid)
    invalid_indices = np.where(np.logical_not(valid))
    valid_items = item[valid_indices]
    invalid_items = item[invalid_indices]
    return valid_items, invalid_items


def filter_valid_dict(valid: tf.Tensor, item: Dict):
    valid_indices = tf.squeeze(tf.where(valid), axis=-1)
    invalid_indices = tf.squeeze(tf.where(tf.logical_not(valid)), axis=-1)
    valid_items = {k: tf.gather(v, valid_indices, axis=0) for k, v in item.items()}
    invalid_items = {k: tf.gather(v, invalid_indices, axis=0) for k, v in item.items()}
    return valid_items, invalid_items


class NewRRT(MyPlanner):

    def __init__(self,
                 fwd_model: BaseDynamicsFunction,
                 filter_model: BaseFilterFunction,
                 classifier_models: List[BaseConstraintChecker],
                 planner_params: Dict,
                 action_params: Dict,
                 scenario: ScenarioWithVisualization,
                 verbose: int,
                 ):
        super().__init__(scenario=scenario, fwd_model=fwd_model, filter_model=filter_model)
        self.verbose = verbose
        self.scenario = scenario
        self.fwd_model = fwd_model
        self.constraint_models = classifier_models
        self.params = planner_params
        self.action_params = action_params

        # These RNGs get re-seeded before planning, so don't bother changing this seed here
        self.state_sampler_rng = rng.Generator.from_seed(0)
        self.goal_sampler_rng = rng.Generator.from_seed(0)
        self.action_sampler_rng = rng.Generator.from_seed(0)
        self.goal_bias_rng = rng.Generator.from_seed(0)
        self.propgation_steps_rng = rng.Generator.from_seed(0)
        self.viz_rng = np.random.RandomState(0)

        self.visualize_propogation_action_color = [0, 0, 0]

        self.batch_size = 256

        self.state_space = DualArmRopeStateSpace(self.scenario, self.params)
        self.action_space = DualArmRopeActionSpace(self.scenario, self.params)

        self.goal_checker = DualArmRopeGoalChecker(self.scenario, self.params, self.batch_size)
        self.goal_sampler = DualArmRopeGoalSampler(self.scenario, self.params, self.goal_sampler_rng)
        self.state_sampler = DualArmRopeStateSampler(self.state_space,
                                                     self.scenario,
                                                     self.params,
                                                     self.state_sampler_rng,
                                                     self.goal_sampler,
                                                     self.batch_size)
        self.action_sampler = DualArmRopeActionSampler(self.action_space,
                                                       self.scenario,
                                                       self.params,
                                                       self.action_params,
                                                       self.action_sampler_rng,
                                                       self.batch_size)

        self.initial_goal_bias = 0.05
        self.max_progagation_steps = self.params.get('max_steps', 10)

        self.set_seeds(0)

    def set_seeds(self, seed):
        self.state_sampler_rng.reset_from_seed(seed)
        self.goal_sampler_rng.reset_from_seed(seed)
        self.action_sampler_rng.reset_from_seed(seed)
        self.goal_bias_rng.reset_from_seed(seed)
        self.propgation_steps_rng.reset_from_seed(seed)
        self.viz_rng = np.random.RandomState(0)

    def plan(self, planning_query: PlanningQuery):
        self.set_seeds(planning_query.seed)

        # create start and goal states
        start_state_tf = {}  # FIXME: this is gross
        for k in self.state_space.sizes.keys():
            if k in planning_query.start:
                start_state_tf[k] = planning_query.start[k]
        start_state_tf['stdev'] = tf.constant([0.0], tf.float32)
        start_state_tf = make_dict_tf_float32(start_state_tf)

        # visualization
        self.scenario.reset_planning_viz()
        self.scenario.plot_environment_rviz(planning_query.environment)
        self.scenario.plot_start_state(planning_query.start)
        self.scenario.plot_goal_rviz(planning_query.goal, self.params['goal_params']['threshold'])

        ptc = TimeoutOrNotProgressing(self, self.params['termination_criteria'], self.verbose)

        # START TIMING
        t0 = perf_counter()

        # actually run the planner
        environment_tf = make_dict_tf_float32(planning_query.environment)
        goal_tf = make_dict_tf_float32(planning_query.goal)
        planning_result = self.plan_impl(environment_tf, start_state_tf, goal_tf, ptc)

        # END TIMING
        planning_time = perf_counter() - t0

        planning_result['time'] = planning_time

        return PlanningResult(**planning_result)

    def plan_impl(self, environment: Dict, start_state: Dict, goal: Dict, ptc):
        planner_status = MyPlannerStatus.Failure
        min_dist_to_goal = 1e10
        valid_next_states_t = None
        valid_selected_indices = None
        selected_indices = None
        tree = MyTree(self.state_space, self.action_space, start_state, self.batch_size)

        environment_batched = repeat(environment, repetitions=self.batch_size, axis=0, new_axis=True)

        start_state_valid = self.is_valid(environment_batched, add_batch(start_state), 1)

        last_lap = perf_counter()
        dts = []

        if start_state_valid:
            while True:
                ptc.condition()
                if ptc.timed_out:
                    planner_status = MyPlannerStatus.Timeout
                    break
                # elif ptc.not_progressing:
                #     planner_status = MyPlannerStatus.NotProgressing
                #     break

                selected_indices, selected_states = tree.sample()

                delta_actions = self.action_sampler.sample()

                now = perf_counter()
                dt = now - last_lap
                dts.append(dt)
                print(".", end='', flush=True)
                last_lap = now

                # FIXME: hack, what's the right way to do this?
                actions = self.action_space.foobar(selected_states, delta_actions)

                next_states_t = self.propagate(environment_batched, selected_states, actions)

                states_valid = self.is_valid(environment_batched, selected_states, self.batch_size)
                transitions_valid, constraint_probabilities = self.check_constraints(environment_batched,
                                                                                     selected_states,
                                                                                     actions,
                                                                                     next_states_t)
                valid = tf.logical_and(states_valid, transitions_valid)

                ptc.attempted_extensions += self.batch_size
                ptc.all_rejected = tf.reduce_any(valid)

                valid_selected_indices, invalid_selected_indices = filter_valid_array(valid, selected_indices)
                valid_next_states_t, invalid_next_states_t = filter_valid_dict(valid, next_states_t)
                valid_actions, invalid_actions = filter_valid_dict(valid, actions)

                if self.verbose >= 1:
                    min_dist_to_goal = self.visualize_nearest_to_goal(goal, min_dist_to_goal, valid_next_states_t)

                    if self.verbose >= 2:
                        self.visualize_propogation(selected_states,
                                                   actions,
                                                   next_states_t,
                                                   valid,
                                                   constraint_probabilities,
                                                   0)

                tree.add(parent_indices=valid_selected_indices, action=valid_actions, state=valid_next_states_t)

                if tf.reduce_any(self.goal_checker.reached(valid_next_states_t, goal)):
                    planner_status = MyPlannerStatus.Solved
                    break  # break out of multi-step propagation

                if planner_status == MyPlannerStatus.Solved:
                    break

        print(f"Mean Iteration Time: {np.mean(dts):.3f}")

        if planner_status == MyPlannerStatus.Solved:
            states, actions = self.extract_states_and_actions(tree, valid_selected_indices, valid_next_states_t, goal)
        elif planner_status == MyPlannerStatus.Timeout:
            states, actions = self.extract_states_and_actions(tree, valid_selected_indices, valid_next_states_t, goal)
        elif planner_status == MyPlannerStatus.NotProgressing:
            states = []
            actions = []
        elif planner_status == MyPlannerStatus.Failure:
            states = []
            actions = []
        else:
            raise NotImplementedError(planner_status)

        return {
            'status':  planner_status,
            'path':    states,
            'actions': actions,
            'tree':    tree,
        }

    def visualize_nearest_to_goal(self, goal, min_dist_to_goal, valid_next_states_t):
        if list(valid_next_states_t.values())[0].shape[0] == 0:
            return min_dist_to_goal

        state_nearest_to_goal, dist_to_goal = self.goal_checker.nearest_to_goal_in_batch(valid_next_states_t, goal)
        if dist_to_goal < min_dist_to_goal:
            min_dist_to_goal = dist_to_goal
            self.scenario.plot_state_closest_to_goal(numpify(state_nearest_to_goal))
        return min_dist_to_goal

    def is_valid(self, environment_batched: Dict, state: Dict, batch_size: int):
        valid = satisfies_bounds(self.params, state, batch_size)
        return valid

    def propagate(self, environment_batched: Dict, states: Dict, actions: Dict) -> Dict:
        # add time dimension
        states = add_batch(states, batch_axis=1)
        actions = add_batch(actions, batch_axis=1)
        mean_pred_states, _ = self.fwd_model.propagate_tf_batched(environment=environment_batched,
                                                                  state=states,
                                                                  actions=actions)
        # get only the final state predicted, since *_predicted_states includes the start state
        next_state = {k: s[:, -1] for k, s in mean_pred_states.items()}

        return next_state

    def check_constraints(self, environment_batched: Dict, state: Dict, action: Dict, next_state: Dict):
        actions = add_batch(action, batch_axis=1)  # add time dimension
        states = sequence_of_dicts_to_dict_of_tensors([state, next_state], axis=1)

        accept = tf.cast(tf.ones(self.batch_size), tf.bool)
        constraint_probabilities = {}
        for constraint_checker in self.constraint_models:
            p_accepts_for_model, _ = constraint_checker.check_constraint_tf_batched(environment=environment_batched,
                                                                                    states=states,
                                                                                    actions=actions,
                                                                                    batch_size=self.batch_size,
                                                                                    state_sequence_length=1)
            p_accept_for_model = p_accepts_for_model[-1]
            constraint_probabilities[constraint_checker.name] = p_accept_for_model
            accept_for_model = p_accept_for_model > self.params['accept_threshold']
            accept = tf.logical_and(accept, accept_for_model)

        return accept, constraint_probabilities

    def visualize_propogation(self,
                              state: Dict,
                              action: Dict,
                              next_state: Dict,
                              valid: tf.Tensor,
                              accept_probabilities: Dict,
                              propagation_step: int
                              ):
        # try to check if this is a new action, in which case we want to sample a new color
        if propagation_step == 0:
            random_color = cm.Dark2(self.viz_rng.uniform(0, 1))
            self.visualize_propogation_action_color = random_color

        for b in range(min(self.batch_size, 10)):
            if 'NNClassifier' in accept_probabilities:
                classifier_probability = accept_probabilities['NNClassifier']
                alpha = min(classifier_probability * 0.8 + 0.2, 0.8)
                state_color = cm.Reds_r(classifier_probability)
            else:
                alpha = 0.8
                state_color = cm.Reds_r(0)

            next_state_b = numpify({k: v[b] for k, v in next_state.items()})
            state_b = numpify({k: v[b] for k, v in state.items()})
            action_b = numpify({k: v[b] for k, v in action.items()})
            if valid[b]:
                self.scenario.plot_tree_state(next_state_b, color=state_color)
                self.scenario.plot_tree_action(state_b,
                                               action_b,
                                               r=self.visualize_propogation_action_color[0],
                                               g=self.visualize_propogation_action_color[1],
                                               b=self.visualize_propogation_action_color[2],
                                               a=alpha)
            else:
                self.scenario.plot_rejected_state(next_state_b)

            self.scenario.plot_current_tree_state(next_state_b, color=state_color)
            self.scenario.plot_current_tree_action(state_b, action_b,
                                                   r=self.visualize_propogation_action_color[0],
                                                   g=self.visualize_propogation_action_color[1],
                                                   b=self.visualize_propogation_action_color[2],
                                                   a=alpha)

    def get_metadata(self):
        return {
            "horizon": self.constraint_models[0].horizon,
        }

    def extract_states_and_actions(self,
                                   tree: MyTree,
                                   valid_selected_indices: np.array,
                                   valid_next_states_t: Dict,
                                   goal: Dict):
        dist_to_goal = self.goal_checker.distance_to_goal(valid_next_states_t, goal)
        nearest_to_goal_idx = tf.argmin(dist_to_goal)
        index_of_state_nearest_goal = valid_selected_indices[nearest_to_goal_idx]
        state_nearest_goal, _ = self.goal_checker.nearest_to_goal_in_batch(valid_next_states_t, goal)

        states = [state_nearest_goal]
        node_index = index_of_state_nearest_goal
        actions = []
        while True:
            action = {k: v[node_index] for k, v in tree.actions.items()}

            node_index = tree.parent_indices[node_index]
            if node_index == MyTree.NO_PARENT:
                break

            actions.append(action)

            state = {k: v[node_index] for k, v in tree.states.items()}
            states.append(state)

        return states, actions
