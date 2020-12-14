from time import perf_counter
from typing import Dict, List, Optional, Callable

import tensorflow as tf
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


class BaseTrajectoryOptimizer:

    def __init__(self,
                 fwd_model: BaseDynamicsFunction,
                 classifier_model: Optional[BaseConstraintChecker],
                 scenario: ExperimentScenario,
                 params: Dict,
                 cost_function: Optional[Callable] = None,
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
        self.override_cost_function = cost_function

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
            actions, planned_path, _ = self.step(environment, goal_state, actions, start_state)

            if self.verbose >= 2:
                self.scenario.plot_state_rviz(numpify(planned_path[1]), label='opt', idx=i)
                self.scenario.plot_action_rviz(numpify(planned_path[0]), numpify(actions[0]), label='opt', idx=i)
        smoothing_time = perf_counter() - start_smoothing_time

        if self.verbose >= 3:
            print("Smoothing time: {:.3f}".format(smoothing_time))

        return actions, planned_path

    def step(self, environment: Dict, goal_state: Dict, actions: List[Dict], start_state: Dict):
        with tf.GradientTape(watch_accessed_variables=True, persistent=True) as tape:
            # Compute the states predicted given the actions
            mean_predictions, _ = self.fwd_model.propagate_differentiable(environment=environment,
                                                                          start_state=start_state,
                                                                          actions=actions)

            if self.override_cost_function is not None:
                loss = self.override_cost_function(actions, environment, goal_state, mean_predictions)
            else:
                loss = self.compute_cost(actions, environment, goal_state, mean_predictions)

        variables = []
        for action in actions:
            for v in action.values():
                variables.append(v)
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        losses = [loss]
        return actions, mean_predictions, losses

    def compute_cost(self, actions: List[Dict], environment: Dict, goal_state: Dict, mean_predictions: List[Dict]):
        # should return loss
        raise NotImplementedError()

    def compute_goal_cost(self, actions, environment, goal_state, mean_predictions):
        # Compute various loss terms
        final_state = mean_predictions[-1]
        goal_loss = self.scenario.trajopt_distance_to_goal_differentiable(final_state, goal_state)
        return goal_loss

    def compute_length_cost(self, actions, environment, goal_state, mean_predictions):
        distances = [self.scenario.trajopt_distance_differentiable(s1, s2) for (s1, s2) in pairwise(mean_predictions)]
        length_loss = tf.reduce_sum(tf.square(distances))
        return length_loss

    def compute_action_cost(self, actions, environment, goal_state, mean_predictions):
        action_loss = self.scenario.trajopt_action_sequence_cost_differentiable(actions)
        return action_loss
