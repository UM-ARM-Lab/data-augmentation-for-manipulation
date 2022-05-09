from time import perf_counter
from typing import Dict, List, Optional, Callable

import torch
from matplotlib import cm
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from moonshine.numpify import numpify
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


def make_torch_parameters(initial_actions):
    def _var(k, a):
        return torch.nn.Parameter(torch.tensor(a))

    out = []
    for initial_action in initial_actions:
        action_variables = {k: _var(k, a) for k, a in initial_action.items()}
        out.append(action_variables)
    return out


def _to_list_of_params(actions: List[Dict]):
    l = []
    for action in actions:
        for v in action.values():
            l.append(v)
    return l


class TrajectoryOptimizerTorch:

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
        self.initial_learning_rate = params['initial_learning_rate']
        self.cost_function = cost_function
        self.optimizer = None

    def optimize(self,
                 environment: Dict,
                 goal_state: Dict,
                 initial_actions: List[Dict],
                 start_state: Dict,
                 ):
        actions = make_torch_parameters(initial_actions)
        self.optimizer = SGD(_to_list_of_params(actions), lr=self.initial_learning_rate)
        lr_decay = ExponentialLR(self.optimizer, gamma=0.95)

        start_smoothing_time = perf_counter()
        planned_path = None
        for i in range(self.iters):
            actions, planned_path, cost = self.step(environment, goal_state, actions, start_state)

            if self.verbose >= 2:
                self.scenario.plot_state_rviz(numpify(planned_path[1]), label='opt', color=cm.Reds(cost), idx=1)
                self.scenario.plot_action_rviz(numpify(planned_path[0]), numpify(actions[0]), label='opt')
        smoothing_time = perf_counter() - start_smoothing_time

        if self.verbose >= 3:
            print("traj follower time: {:.3f}".format(smoothing_time))

        return actions, planned_path

    def step(self, environment: Dict, goal_state: Dict, actions: List[Dict], start_state: Dict):
        self.optimizer.zero_grad()

        mean_predictions, _ = self.fwd_model.propagate_tf(environment=environment,
                                                          start_state=start_state,
                                                          actions=actions)

        cost = self.cost_function(actions, environment, goal_state, mean_predictions)
        cost.backward()
        torch.nn.utils.clip_grad_norm_(_to_list_of_params(actions), 0.1)

        self.optimizer.step()

        # re-run the forward pass now that actions have been updated
        planned_path, _ = self.fwd_model.propagate(environment=environment,
                                                   start_state=start_state,
                                                   actions=actions)

        return actions, planned_path, cost.detach().numpy()
