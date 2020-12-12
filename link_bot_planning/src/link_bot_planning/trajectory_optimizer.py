from time import perf_counter
from typing import Dict, List, Optional

import tensorflow as tf
from more_itertools import pairwise

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_pycommon.experiment_scenario import ExperimentScenario
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
        self.optimizer = tf.keras.optimizers.Adam(params["initial_learning_rate"], amsgrad=True)

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
            actions, planned_path, _, _ = self.step(environment, goal_state, actions, start_state)
        smoothing_time = perf_counter() - start_smoothing_time

        if self.verbose >= 1:
            print("Smoothing time: {:.3f}".format(smoothing_time))

        return actions, planned_path

    def step(self, environment: Dict, goal_state: Dict, actions: List[Dict], start_state: Dict):
        with tf.GradientTape(watch_accessed_variables=True, persistent=True) as tape:
            # Compute the states predicted given the actions
            mean_predictions, _ = self.fwd_model.propagate_differentiable(environment=environment,
                                                                          start_state=start_state,
                                                                          actions=actions)

            length_loss, loss, weighted_losses = self.compute_cost(actions, environment, goal_state, mean_predictions)
            # print(loss)

        variables = []
        for action in actions:
            for v in action.values():
                variables.append(v)
        gradients = tape.gradient(loss, variables)
        # print(gradients)
        self.optimizer.apply_gradients(zip(gradients, variables))

        losses = [weighted_losses[0], weighted_losses[1], weighted_losses[2], weighted_losses[3], loss]
        return actions, mean_predictions, losses, length_loss

    def compute_cost(self, actions, environment, goal_state, mean_predictions):
        # Compute the constraint values predicted given the states and actions
        constraint_costs = self.compute_constraints_cost(environment, actions, mean_predictions)
        constraint_loss = tf.reduce_sum(constraint_costs)
        # Compute various loss terms
        final_state = mean_predictions[-1]
        goal_loss = self.scenario.trajopt_distance_to_goal_differentiable(final_state, goal_state)
        distances = [self.scenario.trajopt_distance_differentiable(s1, s2) for (s1, s2) in pairwise(mean_predictions)]
        length_loss = tf.reduce_sum(tf.square(distances))
        action_loss = self.scenario.trajopt_action_sequence_cost_differentiable(actions)
        losses = tf.convert_to_tensor([length_loss, goal_loss, constraint_loss, action_loss])
        weights = tf.convert_to_tensor([self.length_alpha, self.goal_alpha, self.constraints_alpha, self.action_alpha],
                                       dtype=tf.float32)
        weighted_losses = tf.multiply(losses, weights)
        loss = tf.reduce_sum(weighted_losses, axis=0)
        return length_loss, loss, weighted_losses

    def compute_constraints_cost(self, environment: Dict, actions, predictions):
        if self.classifier_model is None:
            return 0.0

        constraint_costs = []
        for t in range(1, len(predictions)):
            predictions_t = predictions[:t + 1]
            constraint_prediction_t = self.classifier_model.check_constraint_tf(environment=environment,
                                                                                states_sequence=predictions_t,
                                                                                actions=actions)
            # NOTE: this math maps (0 -> 1, 1->0)
            if constraint_prediction_t < 0.5:
                constraint_cost = tf.square(0.5 - constraint_prediction_t)
            else:
                constraint_cost = 0
            constraint_costs.append(constraint_cost)
        return constraint_costs
