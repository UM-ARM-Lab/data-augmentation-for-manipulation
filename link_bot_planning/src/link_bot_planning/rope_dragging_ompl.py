import warnings
from typing import Dict

import numpy as np

from link_bot_pycommon.rope_dragging_scenario import RopeDraggingScenario, rope_key_name
from link_bot_pycommon.scenario_ompl import ScenarioOmpl

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    import ompl.base as ob
    import ompl.control as oc

class RopeDraggingOmpl(ScenarioOmpl):

    def __init__(self, scenario_ompl: RopeDraggingScenario):
        self.s = scenario_ompl

    @staticmethod
    def numpy_to_ompl_state(state_np: Dict, state_out: ob.CompoundState):
        for i in range(3):
            state_out[0][i] = np.float64(state_np['gripper'][i])
        for i in range(RopeDraggingScenario.n_links * 3):
            state_out[1][i] = np.float64(state_np[rope_key_name][i])
        state_out[2][0] = np.float64(state_np['stdev'][0])
        state_out[3][0] = np.float64(state_np['num_diverged'][0])

    @staticmethod
    def ompl_state_to_numpy(ompl_state: ob.CompoundState):
        gripper = np.array([ompl_state[0][0], ompl_state[0][1], ompl_state[0][2]])
        rope = []
        for i in range(RopeDraggingScenario.n_links):
            rope.append(ompl_state[1][3 * i + 0])
            rope.append(ompl_state[1][3 * i + 1])
            rope.append(ompl_state[1][3 * i + 2])
        rope = np.array(rope)
        return {
            'gripper':      gripper,
            rope_key_name:  rope,
            'stdev':        np.array([ompl_state[2][0]]),
            'num_diverged': np.array([ompl_state[3][0]]),
        }

    def ompl_control_to_numpy(self, ompl_state: ob.CompoundState, ompl_control: oc.CompoundControl):
        state_np = self.ompl_state_to_numpy(ompl_state)
        current_gripper_position = state_np['gripper']

        gripper_delta_position = np.array([np.cos(ompl_control[0][0]) * ompl_control[0][1],
                                           np.sin(ompl_control[0][0]) * ompl_control[0][1],
                                           0])
        target_gripper_position = current_gripper_position + gripper_delta_position
        return {
            'gripper_position': target_gripper_position,
        }

    def make_goal_region(self, si: oc.SpaceInformation, rng: np.random.RandomState, params: Dict, goal: Dict,
                         plot: bool):
        return RopeDraggingGoalRegion(si=si,
                                      scenario_ompl=self,
                                      rng=rng,
                                      threshold=params['goal_params']['threshold'],
                                      goal=goal,
                                      plot=plot)

    def make_ompl_state_space(self, planner_params, state_sampler_rng: np.random.RandomState, plot: bool):
        state_space = ob.CompoundStateSpace()

        min_x, max_x, min_y, max_y, min_z, max_z = planner_params['extent']

        gripper_subspace = ob.RealVectorStateSpace(3)
        gripper_bounds = ob.RealVectorBounds(3)
        # these bounds are not used for sampling
        gripper_bounds.setLow(0, min_x)
        gripper_bounds.setHigh(0, max_x)
        gripper_bounds.setLow(1, min_y)
        gripper_bounds.setHigh(1, max_y)
        gripper_bounds.setLow(2, min_z)
        gripper_bounds.setHigh(2, max_z)
        gripper_subspace.setBounds(gripper_bounds)
        gripper_subspace.setName("gripper")
        state_space.addSubspace(gripper_subspace, weight=1)

        rope_subspace = ob.RealVectorStateSpace(RopeDraggingScenario.n_links * 3)
        rope_bounds = ob.RealVectorBounds(RopeDraggingScenario.n_links * 3)
        # these bounds are not used for sampling
        rope_bounds.setLow(-1000)
        rope_bounds.setHigh(1000)
        rope_subspace.setBounds(rope_bounds)
        rope_subspace.setName("rope")
        state_space.addSubspace(rope_subspace, weight=1)

        # extra subspace component for the variance, which is necessary to pass information from propagate to constraint checker
        stdev_subspace = ob.RealVectorStateSpace(1)
        stdev_bounds = ob.RealVectorBounds(1)
        stdev_bounds.setLow(-1000)
        stdev_bounds.setHigh(1000)
        stdev_subspace.setBounds(stdev_bounds)
        stdev_subspace.setName("stdev")
        state_space.addSubspace(stdev_subspace, weight=0)

        # extra subspace component for the number of diverged steps
        num_diverged_subspace = ob.RealVectorStateSpace(1)
        num_diverged_bounds = ob.RealVectorBounds(1)
        num_diverged_bounds.setLow(-1000)
        num_diverged_bounds.setHigh(1000)
        num_diverged_subspace.setBounds(num_diverged_bounds)
        num_diverged_subspace.setName("stdev")
        state_space.addSubspace(num_diverged_subspace, weight=0)

        def _state_sampler_allocator(state_space):
            return RopeDraggingStateSampler(state_space,
                                            scenario_ompl=self,
                                            extent=planner_params['extent'],
                                            rng=state_sampler_rng,
                                            plot=plot)

        state_space.setStateSamplerAllocator(ob.StateSamplerAllocator(_state_sampler_allocator))

        return state_space

    def make_ompl_control_space(self, state_space, rng: np.random.RandomState, action_params: Dict):
        control_space = oc.CompoundControlSpace(state_space)

        gripper_control_space = oc.RealVectorControlSpace(state_space, 3)
        gripper_control_bounds = ob.RealVectorBounds(3)
        # Direction (in XY plane)
        gripper_control_bounds.setLow(1, -np.pi)
        gripper_control_bounds.setHigh(1, np.pi)
        # Displacement
        self.action_params = action_params  # FIXME: terrible API
        max_d = action_params['max_distance_gripper_can_move']
        gripper_control_bounds.setLow(2, 0)
        gripper_control_bounds.setHigh(2, max_d)
        gripper_control_space.setBounds(gripper_control_bounds)
        control_space.addSubspace(gripper_control_space)

        def _allocator(cs):
            return RopeDraggingControlSampler(cs, scenario_ompl=self, rng=rng, action_params=action_params)

        # I override the sampler here so I can use numpy RNG to make things more deterministic.
        # ompl does not allow resetting of seeds, which causes problems when evaluating multiple
        # planning queries in a row.
        control_space.setControlSamplerAllocator(oc.ControlSamplerAllocator(_allocator))

        return control_space


class RopeDraggingControlSampler(oc.ControlSampler):
    def __init__(self,
                 control_space: oc.CompoundControlSpace,
                 scenario_ompl: RopeDraggingOmpl,
                 rng: np.random.RandomState,
                 action_params: Dict):
        super().__init__(control_space)
        self.scenario_ompl = scenario_ompl
        self.rng = rng
        self.control_space = control_space
        self.action_params = action_params

    def sampleNext(self, control_out, previous_control, state):
        # Direction
        yaw = self.rng.uniform(-np.pi, np.pi)
        # Displacement
        displacement = self.rng.uniform(0, self.action_params['max_distance_gripper_can_move'])

        control_out[0][0] = yaw
        control_out[0][1] = displacement

    def sampleStepCount(self, min_steps, max_steps):
        step_count = self.rng.randint(min_steps, max_steps)
        return step_count


class RopeDraggingStateSampler(ob.RealVectorStateSampler):

    def __init__(self,
                 state_space,
                 scenario_ompl: RopeDraggingOmpl,
                 extent,
                 rng: np.random.RandomState,
                 plot: bool):
        super().__init__(state_space)
        self.scenario_ompl = scenario_ompl
        self.extent = np.array(extent).reshape(3, 2)
        self.rng = rng
        self.plot = plot

    def sampleUniform(self, state_out: ob.CompoundState):
        # trying to sample a "valid" rope state is difficult, and probably unimportant
        # because the only role this plays in planning is to cause exploration/expansion
        # by biasing towards regions of empty space. So here we just pick a random point
        # and duplicate it, as if all points on the rope were at this point
        random_point = self.rng.uniform(self.extent[:, 0], self.extent[:, 1])
        random_point_rope = np.concatenate([random_point] * RopeDraggingScenario.n_links)
        state_np = {
            'gripper':      random_point,
            rope_key_name:  random_point_rope,
            'num_diverged': np.zeros(1, dtype=np.float64),
            'stdev':        np.zeros(1, dtype=np.float64),
        }

        self.scenario_ompl.numpy_to_ompl_state(state_np, state_out)

        if self.plot:
            self.scenario_ompl.s.plot_sampled_state(state_np)


class RopeDraggingGoalRegion(ob.GoalSampleableRegion):

    def __init__(self,
                 si: oc.SpaceInformation,
                 scenario_ompl: RopeDraggingOmpl,
                 rng: np.random.RandomState,
                 threshold: float,
                 goal: Dict,
                 plot: bool):
        super(RopeDraggingGoalRegion, self).__init__(si)
        self.setThreshold(threshold)
        self.goal = goal
        self.scenario_ompl = scenario_ompl
        self.rng = rng
        self.plot = plot

    def distanceGoal(self, state: ob.CompoundState):
        """
        Uses the distance between a specific point in a specific subspace and the goal point
        """
        state_np = self.scenario_ompl.ompl_state_to_numpy(state)
        distance = self.scenario_ompl.s.distance_to_goal(state_np, self.goal)

        # this ensures the goal must have num_diverged = 0
        if state_np['num_diverged'] > 0:
            distance = 1e9
        return distance

    def sampleGoal(self, state_out: ob.CompoundState):
        sampler = self.getSpaceInformation().allocStateSampler()
        # sample a random state via the state space sampler, in hopes that OMPL will clean up the memory...
        sampler.sampleUniform(state_out)

        # don't bother trying to sample "legit" rope states, because this is only used to bias sampling towards the goal
        # so just prenteing every point on therope is at the goal should be sufficient
        rope = np.concatenate([self.goal['tail']] * RopeDraggingScenario.n_links)

        goal_state_np = {
            'gripper':      self.goal['tail'],
            rope_key_name:  rope,
            'num_diverged': np.zeros(1, dtype=np.float64),
            'stdev':        np.zeros(1, dtype=np.float64),
        }

        self.scenario_ompl.numpy_to_ompl_state(goal_state_np, state_out)

        if self.plot:
            self.scenario_ompl.s.plot_sampled_goal_state(goal_state_np)

    def maxSampleCount(self):
        return 100
