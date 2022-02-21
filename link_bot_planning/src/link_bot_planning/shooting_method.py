from time import perf_counter
from typing import Optional, Dict

import numpy as np
import tensorflow as tf
from matplotlib import cm

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_planning.my_planner import MyPlannerStatus, PlanningResult, MyPlanner, PlanningQuery
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from moonshine.moonshine_utils import dict_of_sequences_to_sequence_of_dicts
from moonshine.tensorflow_utils import sequence_of_dicts_to_dict_of_tensors
from moonshine.numpify import numpify
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction
from state_space_dynamics.base_filter_function import BaseFilterFunction, PassThroughFilter


class ShootingMethod(MyPlanner):

    def __init__(self,
                 fwd_model: BaseDynamicsFunction,
                 classifier_model: Optional[BaseConstraintChecker],
                 scenario: ExperimentScenario,
                 params: Dict,
                 action_params: Dict,
                 filter_model: BaseFilterFunction = PassThroughFilter(),
                 verbose: Optional[int] = 0):
        super().__init__(scenario=scenario,
                         fwd_model=fwd_model,
                         filter_model=filter_model,
                         decoder=None)
        self.fwd_model = fwd_model
        self.classifier_model = classifier_model
        self.verbose = verbose
        self.scenario = scenario
        self.n_samples = params["n_samples"]
        self.action_params = action_params

    def plan_internal(self, planning_query: PlanningQuery) -> PlanningResult:
        start_planning_time = perf_counter()

        action_rng = np.random.RandomState(planning_query.seed)
        random_actions = self.scenario.sample_action_sequences(environment=planning_query.environment,
                                                               state=planning_query.start,
                                                               action_params=self.action_params,
                                                               n_action_sequences=self.n_samples,
                                                               action_sequence_length=1,
                                                               validate=True,
                                                               action_rng=action_rng)
        random_actions = [sequence_of_dicts_to_dict_of_tensors(a) for a in random_actions]
        random_actions = sequence_of_dicts_to_dict_of_tensors(random_actions)

        environment_batched = {k: tf.stack([v] * self.n_samples, axis=0) for k, v in planning_query.environment.items()}
        start_state_batched = {k: tf.expand_dims(tf.stack([v] * self.n_samples, axis=0), axis=1) for k, v in
                               planning_query.start.items()}
        mean_predictions, _ = self.fwd_model.propagate_tf_batched(environment=environment_batched,
                                                                  state=start_state_batched,
                                                                  actions=random_actions)

        final_states = {k: v[:, -1] for k, v in mean_predictions.items()}
        goal_state_batched = {k: tf.stack([v] * self.n_samples, axis=0) for k, v in planning_query.goal.items()}
        costs = self.scenario.trajopt_distance_to_goal_differentiable(final_states, goal_state_batched)
        costs = tf.squeeze(costs)
        min_idx = tf.math.argmin(costs, axis=0)
        # print(costs)
        # print('min idx', min_idx.numpy())
        best_indices = tf.argsort(costs)

        cmap = cm.Blues
        n_to_show = 5
        min_cost = costs[best_indices[0]]
        max_cost = costs[best_indices[n_to_show]]
        for j, i in enumerate(best_indices[:n_to_show]):
            s = numpify({k: v[i] for k, v in start_state_batched.items()})
            a = numpify({k: v[i][0] for k, v in random_actions.items()})
            c = (costs[i] - min_cost) / (max_cost - min_cost)
            self.scenario.plot_action_rviz(s, a, label='samples', color=cmap(c), idx1=2 * j, idx2=2 * j + 1)

        best_actions = {k: v[min_idx] for k, v in random_actions.items()}
        best_prediction = {k: v[min_idx] for k, v in mean_predictions.items()}

        best_actions = numpify(dict_of_sequences_to_sequence_of_dicts(best_actions))
        best_predictions = numpify(dict_of_sequences_to_sequence_of_dicts(best_prediction))

        planning_time = perf_counter() - start_planning_time
        planner_status = MyPlannerStatus.Solved
        return PlanningResult(status=planner_status, path=best_predictions, actions=best_actions, time=planning_time, tree={})

    def get_metadata(self):
        return {
            "n_samples": self.n_samples
        }
