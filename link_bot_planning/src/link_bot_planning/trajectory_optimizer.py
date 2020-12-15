from time import perf_counter
from typing import Dict, List, Optional, Callable

import tensorflow as tf
from matplotlib import cm
from more_itertools import pairwise

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from moonshine.moonshine_utils import numpify
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


def make_tf_variables(initial_actions):
    def _var(k, a):
        return tf.Variable(a, dtype=tf.float32, name=k, trainable=True)

    out = []
    for initial_action in initial_actions:
        action_variables = {k: _var(k, a) for k, a in initial_action.items()}
        out.append(action_variables)
    return out


class TrajectoryOptimizer:

    def __init__(self,
                 fwd_model: BaseDynamicsFunction,
                 classifier_model: Optional[BaseConstraintChecker],
                 scenario: ExperimentScenario,
                 params: Dict,
                 cost_function: Callable,
                 verbose: Optional[int] = 0,
                 ):
        self.fwd_model = fwd_model
        self.classifier_model = classifier_model
        self.verbose = verbose
        self.scenario = scenario
        self.iters = params["iters"]
        self.length_alpha = params["length_alpha"]
        self.goal_alpha = params["goal_alpha"]
        self.constraints_alpha = params["constraints_alpha"]
        self.action_alpha = params["action_alpha"]

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(params["initial_learning_rate"],
                                                                     decay_steps=1,
                                                                     decay_rate=0.95,
                                                                     staircase=True)

        self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
        self.cost_function = cost_function

    def optimize(self,
                 environment: Dict,
                 goal_state: Dict,
                 initial_actions: List[Dict],
                 start_state: Dict,
                 ):
        actions = make_tf_variables(initial_actions)

        start_smoothing_time = perf_counter()
        planned_path = None
        for i in range(self.iters):
            actions, planned_path, cost = self.step(environment, goal_state, actions, start_state)

            if self.verbose >= 2:
                self.scenario.plot_state_rviz(numpify(planned_path[1]), label='opt', idx=i, color=cm.Reds(cost))
                self.scenario.plot_action_rviz(numpify(planned_path[0]), numpify(actions[0]), label='opt', idx=i)
        smoothing_time = perf_counter() - start_smoothing_time

        if self.verbose >= 3:
            print("traj follower time: {:.3f}".format(smoothing_time))

        return actions, planned_path

    def step(self, environment: Dict, goal_state: Dict, actions: List[Dict], start_state: Dict):
        with tf.GradientTape(watch_accessed_variables=True, persistent=True) as tape:
            # Compute the states predicted given the actions
            mean_predictions, _ = self.fwd_model.propagate_differentiable(environment=environment,
                                                                          start_state=start_state,
                                                                          actions=actions)

            cost = self.cost_function(actions, environment, goal_state, mean_predictions)

        variables = []
        for action in actions:
            for v in action.values():
                variables.append(v)
        gradients = tape.gradient(cost, variables)

        # g can be None if there are parts of the network not being trained, i.e. the observer with there are no obs. feats.
        valid_grads_and_vars = [(g, v) for (g, v) in zip(gradients, variables) if g is not None]
        # clip for stability
        valid_grads_and_vars = [(tf.clip_by_value(g, -0.1, 0.1), v) for (g, v) in valid_grads_and_vars]

        self.optimizer.apply_gradients(valid_grads_and_vars)

        return actions, mean_predictions, cost


def compute_goal_cost(scenario, goal_state, mean_predictions):
    # Compute various cost terms
    final_state = mean_predictions[-1]
    goal_cost = scenario.trajopt_distance_to_goal_differentiable(final_state, goal_state)
    return goal_cost


def compute_length_cost(scenario, mean_predictions):
    distances = [scenario.trajopt_distance_differentiable(s1, s2) for (s1, s2) in pairwise(mean_predictions)]
    length_cost = tf.reduce_sum(tf.square(distances))
    return length_cost


def compute_action_cost(scenario, actions):
    action_cost = scenario.trajopt_action_sequence_cost_differentiable(actions)
    return action_cost


def compute_constraints_cost(classifier_model, actions: List, environment: Dict, states: List[Dict]):
    constraint_costs = []
    for t in range(1, len(states)):
        s_t = states[:t + 1]
        accept_probability_t, _ = classifier_model.check_constraint_tf(environment=environment,
                                                                       states_sequence=s_t,
                                                                       actions=actions)
        accept_probability_t = accept_probability_t[0]
        if accept_probability_t > 0.5:
            constraint_cost = tf.zeros_like(accept_probability_t)
        else:
            constraint_cost = tf.square(0.5 - accept_probability_t)
        constraint_costs.append(constraint_cost)
    return constraint_costs
