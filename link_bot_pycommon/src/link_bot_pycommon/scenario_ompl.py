import warnings
from typing import Dict

from numpy.random import RandomState

from link_bot_planning.my_planner import SharedPlanningStateOMPL
from link_bot_planning.trajectory_optimizer import TrajectoryOptimizer
from link_bot_pycommon.experiment_scenario import ExperimentScenario

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    import ompl.base as ob
    import ompl.control as oc


class ScenarioOmpl:

    def __init__(self,
                 scenario: ExperimentScenario,
                 planner_params: Dict,
                 action_params: Dict,
                 state_sampler_rng: RandomState,
                 control_sampler_rng: RandomState,
                 shared_planning_state: SharedPlanningStateOMPL,
                 plot: bool,
                 ):
        self.s = scenario
        self.planner_params = planner_params
        self.action_params = action_params
        self.state_sampler_rng = state_sampler_rng
        self.control_sampler_rng = control_sampler_rng
        self.plot = plot
        self.sps = shared_planning_state
        self.state_space = self.make_state_space()
        self.control_space = self.make_control_space()

    def numpy_to_ompl_state(self, state_np: Dict, state_out: ob.CompoundState):
        raise NotImplementedError()

    def numpy_to_ompl_control(self, state_np: Dict, control_np: Dict, control_out: oc.CompoundControl):
        raise NotImplementedError()

    def ompl_state_to_numpy(self, ompl_state: ob.CompoundState):
        raise NotImplementedError()

    def ompl_control_to_numpy(self, ompl_state: ob.CompoundState, ompl_control: oc.CompoundControl):
        raise NotImplementedError()

    def make_goal_region(self,
                         si: oc.SpaceInformation,
                         rng: RandomState,
                         params: Dict,
                         goal: Dict,
                         plot: bool):
        raise NotImplementedError()

    def make_state_space(self):
        raise NotImplementedError()

    def make_control_space(self):
        raise NotImplementedError()

    def make_directed_control_sampler(self,
                                      si: oc.SpaceInformation,
                                      rng: RandomState,
                                      action_params: Dict,
                                      opt: TrajectoryOptimizer,
                                      max_steps: int):
        raise NotImplementedError()
