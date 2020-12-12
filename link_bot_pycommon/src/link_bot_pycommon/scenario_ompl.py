import warnings
from typing import Dict

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    import ompl.base as ob
    import ompl.control as oc


class ScenarioOmpl:

    @staticmethod
    def numpy_to_ompl_state(state_np: Dict, state_out: ob.CompoundState):
        raise NotImplementedError()

    @staticmethod
    def ompl_state_to_numpy(ompl_state: ob.CompoundState):
        raise NotImplementedError()

    def ompl_control_to_numpy(self, ompl_state: ob.CompoundState, ompl_control: oc.CompoundControl):
        raise NotImplementedError()

    def make_goal_region(self, si: oc.SpaceInformation, rng: np.random.RandomState, params: Dict, goal: Dict,
                         plot: bool):
        raise NotImplementedError()

    def make_ompl_state_space(self, planner_params, state_sampler_rng: np.random.RandomState, plot: bool):
        raise NotImplementedError()

    def make_ompl_control_space(self, state_space, rng: np.random.RandomState, action_params: Dict):
        raise NotImplementedError()

    def make_directed_control_sampler(self, si: oc.SpaceInformation):
        raise NotImplementedError()
