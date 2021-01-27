from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, List

from dataclasses_json import dataclass_json

from link_bot_planning.base_decoder_function import BaseDecoderFunction, PassThroughDecoderFunction
from link_bot_pycommon.animatable_scenario import AnimatableScenario
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction
from state_space_dynamics.base_filter_function import BaseFilterFunction, PassThroughFilter


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


@dataclass_json
@dataclass
class PlanningQuery:
    goal: Dict
    environment: Dict
    start: Dict
    seed: int


@dataclass_json
@dataclass
class PlanningResult:
    path: Optional[List[Dict]]
    actions: Optional[List[Dict]]
    status: MyPlannerStatus
    tree: Dict
    time: float


class MyPlanner:
    def __init__(self,
                 scenario: AnimatableScenario,
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
                                              seed=planning_query.seed)
        planning_result = self.plan_internal(planning_query=latent_planning_query)

        return planning_result

    def plan_internal(self, planning_query: PlanningQuery) -> PlanningResult:
        raise NotImplementedError()

    def get_metadata(self):
        return {}
