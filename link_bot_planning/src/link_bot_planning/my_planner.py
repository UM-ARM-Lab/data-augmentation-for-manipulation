import pathlib
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, List

import numpy as np
from dataclasses_json import dataclass_json

from link_bot_planning.base_decoder_function import BaseDecoderFunction, PassThroughDecoderFunction
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction
from state_space_dynamics.base_filter_function import BaseFilterFunction, PassThroughFilter


def are_states_close(a: Dict, b: Dict):
    # assert (set(a.keys()) == set(b.keys()))
    for k in ['rope', 'left_gripper', 'right_gripper']:
        v1 = a[k]
        v2 = b[k]
        if isinstance(v1, np.ndarray):
            if v1.dtype in [np.float32, np.float64, np.int32, np.int64]:
                if not np.allclose(v1, v2):
                    return False
    return True


class MyPlannerStatus(Enum):
    Solved = "solved"
    Timeout = "timeout"
    Failure = "failure"
    NotProgressing = "not progressing"

    def __bool__(self):
        if self.value == MyPlannerStatus.Solved:
            return True
        elif self.value == MyPlannerStatus.Timeout:
            return True
        else:
            return False


class LoggingTree:
    """
    This duplicates what OMPL does already, but the OMPL implementation is not python friendly
    """

    def __init__(self, state=None, action=None, accept_probabilities=None):
        self.state = state
        self.action = action
        self.children: List[LoggingTree] = []
        self.accept_probabilities = accept_probabilities
        self.size = 0

        self.cached = self

    def add(self, before_state: Dict, action: Dict, after_state: Dict, accept_probabilities):
        self.size += 1
        if len(self.children) == 0:
            self.state = before_state
            t = self
        else:
            # perf optimization
            if are_states_close(self.cached.state, before_state):
                t = self.cached
            else:
                t = self.find(before_state)

        new_child = LoggingTree(state=after_state, action=action, accept_probabilities=accept_probabilities)
        t.children.append(new_child)
        self.cached = new_child

    def find(self, state: Dict):
        if are_states_close(self.state, state):
            return self
        for child in self.children:
            s = child.find(state)
            if s is not None:
                return s
        return None

    def __str__(self):
        s = ""
        for child in self.children:
            s += child.__str__()
        s += str(self.state)
        return s


@dataclass_json
@dataclass
class SetupInfo:
    bagfile_name: Optional[pathlib.Path]


@dataclass_json
@dataclass
class PlanningQuery:
    goal: Dict
    environment: Dict
    start: Dict
    seed: int
    trial_start_time_seconds: float  # seconds, based on perf_counter()


@dataclass
class PlanningResult:
    path: Optional[List[Dict]]
    actions: Optional[List[Dict]]
    status: MyPlannerStatus
    tree: LoggingTree
    time: float
    mean_propagate_time: float = None


@dataclass
class SharedPlanningStateOMPL:
    environment: Optional[Dict] = None
    just_sampled_goal: bool = False
    just_sampled_new_action: bool = False


class MyPlanner:
    def __init__(self,
                 scenario: ScenarioWithVisualization,
                 fwd_model: BaseDynamicsFunction,
                 filter_model: BaseFilterFunction = PassThroughFilter(),
                 decoder: Optional[BaseDecoderFunction] = PassThroughDecoderFunction()):
        self.decoder = decoder
        self.scenario = scenario
        self.fwd_model = fwd_model
        self.filter_model = filter_model

    def plan(self, planning_query: PlanningQuery) -> PlanningResult:
        mean_start, _ = self.filter_model.filter(environment=planning_query.environment,
                                                 state=None,
                                                 observation=planning_query.start)
        mean_goal, _ = self.filter_model.filter(environment=planning_query.environment,
                                                state=None,
                                                observation=planning_query.goal)
        latent_planning_query = PlanningQuery(start=mean_start,
                                              goal=mean_goal,
                                              environment=planning_query.environment,
                                              seed=planning_query.seed,
                                              trial_start_time_seconds=planning_query.trial_start_time_seconds)
        planning_result = self.plan_internal(planning_query=latent_planning_query)

        return planning_result

    def plan_internal(self, planning_query: PlanningQuery) -> PlanningResult:
        raise NotImplementedError()

    def get_metadata(self):
        return {}
