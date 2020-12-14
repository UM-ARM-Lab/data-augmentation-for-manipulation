from typing import Dict, List, Optional, Callable

import tensorflow as tf

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_planning.base_trajectory_optimizer import BaseTrajectoryOptimizer
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


class TrajectoryOptimizer(BaseTrajectoryOptimizer):

    def __init__(self,
                 fwd_model: BaseDynamicsFunction,
                 classifier_model: Optional[BaseConstraintChecker],
                 scenario: ExperimentScenario,
                 params: Dict,
                 cost_function: Optional[Callable] = None,
                 verbose: Optional[int] = 0,
                 ):
        super().__init__(fwd_model=fwd_model,
                         classifier_model=classifier_model,
                         scenario=scenario,
                         params=params,
                         cost_function=cost_function,
                         verbose=verbose)

    def compute_cost(self, actions: List[Dict], environment: Dict, goal_state: Dict, mean_predictions: List[Dict]):
        # constraint_loss = self.compute_constraints_cost(actions, environment, goal_state, mean_predictions)
        goal_loss = self.compute_goal_cost(actions, environment, goal_state, mean_predictions)
        # length_loss = self.compute_length_cost(actions, environment, goal_state, mean_predictions)
        # action_loss = self.compute_action_cost(actions, environment, goal_state, mean_predictions)

        losses = tf.convert_to_tensor([goal_loss])
        weights = tf.convert_to_tensor([self.goal_alpha], dtype=tf.float32)
        weighted_losses = tf.multiply(losses, weights)
        loss = tf.reduce_sum(weighted_losses, axis=0)

        return loss

    def compute_constraints_cost(self, actions, environment, goal_state, predictions):
        del goal_state  # unused

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
