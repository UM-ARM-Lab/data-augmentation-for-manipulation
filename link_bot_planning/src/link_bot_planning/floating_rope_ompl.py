import warnings
from typing import Dict

import numpy as np

from arc_utilities.transformation_helper import vector3_to_spherical, spherical_to_vector3
from link_bot_planning.trajectory_optimizer import TrajectoryOptimizer
from link_bot_pycommon.floating_rope_scenario import FloatingRopeScenario
from link_bot_pycommon.scenario_ompl import ScenarioOmpl
from moonshine.moonshine_utils import numpify
from tf import transformations

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    import ompl.base as ob
    import ompl.control as oc

import rospy
from jsk_recognition_msgs.msg import BoundingBox
from link_bot_pycommon.bbox_visualization import extent_to_bbox


def sample_rope_and_grippers(rng, g1, g2, p, n_links, kd):
    g1 = g1 + rng.uniform(-0.05, 0.05, 3)
    g2 = g2 + rng.uniform(-0.05, 0.05, 3)
    p = p + rng.uniform(-0.05, 0.05, 3)
    n_exclude = 5
    k = rng.randint(n_exclude, n_links + 1 - n_exclude)
    rope = [g2]
    for i in range(1, k - 1):
        new_p = (p - g2) * (i / (k - 1))
        noise = rng.uniform([-kd, -kd, -kd], [kd, kd, kd], 3)
        new_p = g2 + new_p + noise
        rope.append(new_p)
    rope.append(p)
    for i in range(1, n_links - k + 1):
        new_p = (g1 - p) * i / (n_links - k)
        noise = rng.uniform([-kd, -kd, -kd], [kd, kd, kd], 3)
        new_p = p + new_p + noise
        rope.append(new_p)
    rope = np.array(rope)
    return rope


def sample_rope_grippers(rng, g1, g2, n_links):
    rope = [g2 + rng.uniform(-0.01, 0.01, 3)]
    for _ in range(n_links - 2):
        xmin = min(g1[0], g2[0]) - 0.1
        ymin = min(g1[1], g2[1]) - 0.1
        zmin = min(g1[2], g2[2]) - 0.4
        xmax = max(g1[0], g2[0]) + 0.1
        ymax = max(g1[1], g2[1]) + 0.1
        zmax = max(g1[2], g2[2]) + 0.01
        p = rng.uniform([xmin, ymin, zmin], [xmax, ymax, zmax])
        rope.append(p)
    rope.append(g1 + rng.uniform(-0.01, 0.01, 3))
    return np.array(rope)


def sample_rope(rng, p, n_links, kd: float):
    p = np.array(p, dtype=np.float32)
    n_exclude = 5
    k = rng.randint(n_exclude, n_links + 1 - n_exclude)
    # the kth point of the rope is put at the point p
    rope = [p]
    previous_point = np.copy(p)
    for i in range(0, k):
        noise = rng.uniform([-kd, -kd, -kd / 2], [kd, kd, kd * 1.2], 3)
        previous_point = previous_point + noise
        rope.insert(0, previous_point)
    next_point = np.copy(p)
    for i in range(k, n_links):
        noise = rng.uniform([-kd, -kd, -kd / 2], [kd, kd, kd * 1.2], 3)
        next_point = next_point + noise
        rope.append(next_point)
    rope = np.array(rope)
    return rope


class FloatingRopeOmpl(ScenarioOmpl):

    def __init__(self, scenario: FloatingRopeScenario, *args, **kwargs):
        super().__init__(scenario, *args, **kwargs)

    def numpy_to_ompl_state(self, state_np: Dict, state_out: ob.CompoundState):
        rope_points = np.reshape(state_np['rope'], [-1, 3])
        for i in range(3):
            state_out[0][i] = np.float64(state_np['left_gripper'][i])
        for i in range(3):
            state_out[1][i] = np.float64(state_np['right_gripper'][i])
        for j in range(FloatingRopeScenario.n_links):
            for i in range(3):
                state_out[2 + j][i] = np.float64(rope_points[j][i])
        state_out[FloatingRopeScenario.n_links + 2][0] = np.float64(state_np['stdev'][0])
        state_out[FloatingRopeScenario.n_links + 3][0] = np.float64(state_np['num_diverged'][0])

    def numpy_to_ompl_control(self, state_np: Dict, control_np: Dict, control_out: oc.CompoundControl):
        left_gripper_delta = control_np['left_gripper_position'] - state_np['left_gripper']
        left_r, left_phi, left_theta = vector3_to_spherical(left_gripper_delta)

        right_gripper_delta = control_np['right_gripper_position'] - state_np['right_gripper']
        right_r, right_phi, right_theta = vector3_to_spherical(right_gripper_delta)

        control_out[0][0] = np.float_(left_r)
        control_out[0][1] = np.float_(left_phi)
        control_out[0][2] = np.float_(left_theta)

        control_out[1][0] = np.float_(right_r)
        control_out[1][1] = np.float_(right_phi)
        control_out[1][2] = np.float_(right_theta)

    def ompl_state_to_numpy(self, ompl_state: ob.CompoundState):
        left_gripper = np.array([ompl_state[0][0], ompl_state[0][1], ompl_state[0][2]], np.float32)
        right_gripper = np.array([ompl_state[1][0], ompl_state[1][1], ompl_state[1][2]], np.float32)
        rope = []
        for i in range(FloatingRopeScenario.n_links):
            rope.append(ompl_state[2 + i][0])
            rope.append(ompl_state[2 + i][1])
            rope.append(ompl_state[2 + i][2])
        rope = np.array(rope, np.float32)
        return {
            'left_gripper':  left_gripper,
            'right_gripper': right_gripper,
            'rope':          rope,
            'stdev':         np.array([ompl_state[FloatingRopeScenario.n_links + 2][0]], np.float32),
            'num_diverged':  np.array([ompl_state[FloatingRopeScenario.n_links + 3][0]], np.float32),
        }

    def ompl_control_to_numpy(self, ompl_state: ob.CompoundState, ompl_control: oc.CompoundControl):
        state_np = self.ompl_state_to_numpy(ompl_state)
        current_left_gripper_position = state_np['left_gripper']
        current_right_gripper_position = state_np['right_gripper']

        left_gripper_delta_position = spherical_to_vector3([ompl_control[0][0],
                                                            ompl_control[0][1],
                                                            ompl_control[0][2]])

        right_gripper_delta_position = spherical_to_vector3([ompl_control[1][0],
                                                             ompl_control[1][1],
                                                             ompl_control[1][2]])

        target_left_gripper_position = current_left_gripper_position + left_gripper_delta_position
        target_right_gripper_position = current_right_gripper_position + right_gripper_delta_position
        return {
            'left_gripper_position':  target_left_gripper_position,
            'right_gripper_position': target_right_gripper_position,
        }

    def make_goal_region(self,
                         si: oc.SpaceInformation,
                         rng: np.random.RandomState,
                         params: Dict, goal: Dict,
                         plot: bool):
        if goal['goal_type'] == 'midpoint':
            return RopeMidpointGoalRegion(si=si,
                                          scenario_ompl=self,
                                          rng=rng,
                                          threshold=params['goal_params']['threshold'],
                                          goal=goal,
                                          plot=plot)
        elif goal['goal_type'] == 'any_point':
            return RopeAnyPointGoalRegion(si=si,
                                          scenario_ompl=self,
                                          rng=rng,
                                          threshold=params['goal_params']['threshold'],
                                          goal=goal,
                                          plot=plot)
        elif goal['goal_type'] == 'grippers':
            return DualGripperGoalRegion(si=si,
                                         scenario_ompl=self,
                                         rng=rng,
                                         threshold=params['goal_params']['threshold'],
                                         goal=goal,
                                         plot=plot)
        elif goal['goal_type'] == 'grippers_and_point':
            return RopeAndGrippersGoalRegion(si=si,
                                             scenario_ompl=self,
                                             rng=rng,
                                             threshold=params['goal_params']['threshold'],
                                             goal=goal,
                                             plot=plot)
        else:
            raise NotImplementedError()

    def make_state_space(self, planner_params, state_sampler_rng: np.random.RandomState,
                         plot: bool):
        state_space = ob.CompoundStateSpace()

        min_x, max_x, min_y, max_y, min_z, max_z = planner_params['extent']

        left_gripper_subspace = ob.RealVectorStateSpace(3)
        left_gripper_bounds = ob.RealVectorBounds(3)
        left_gripper_bounds.setLow(0, min_x)
        left_gripper_bounds.setHigh(0, max_x)
        left_gripper_bounds.setLow(1, min_y)
        left_gripper_bounds.setHigh(1, max_y)
        left_gripper_bounds.setLow(2, min_z)
        left_gripper_bounds.setHigh(2, max_z)
        left_gripper_subspace.setBounds(left_gripper_bounds)
        left_gripper_subspace.setName("left_gripper")
        state_space.addSubspace(left_gripper_subspace, weight=1)

        right_gripper_subspace = ob.RealVectorStateSpace(3)
        right_gripper_bounds = ob.RealVectorBounds(3)
        right_gripper_bounds.setLow(0, min_x)
        right_gripper_bounds.setHigh(0, max_x)
        right_gripper_bounds.setLow(1, min_y)
        right_gripper_bounds.setHigh(1, max_y)
        right_gripper_bounds.setLow(2, min_z)
        right_gripper_bounds.setHigh(2, max_z)
        right_gripper_subspace.setBounds(right_gripper_bounds)
        right_gripper_subspace.setName("right_gripper")
        state_space.addSubspace(right_gripper_subspace, weight=1)

        for i in range(FloatingRopeScenario.n_links):
            rope_point_subspace = ob.RealVectorStateSpace(3)
            rope_point_bounds = ob.RealVectorBounds(3)
            rope_point_bounds.setLow(0, min_x)
            rope_point_bounds.setHigh(0, max_x)
            rope_point_bounds.setLow(1, min_y)
            rope_point_bounds.setHigh(1, max_y)
            rope_point_bounds.setLow(2, min_z)
            rope_point_bounds.setHigh(2, max_z)
            rope_point_subspace.setBounds(rope_point_bounds)
            rope_point_subspace.setName(f"rope_{i}")
            state_space.addSubspace(rope_point_subspace, weight=1)

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
            return DualGripperStateSampler(state_space,
                                           scenario_ompl=self,
                                           extent=planner_params['state_sampler_extent'],
                                           rng=state_sampler_rng,
                                           plot=plot)

        state_space.setStateSamplerAllocator(ob.StateSamplerAllocator(_state_sampler_allocator))

        return state_space

    def make_control_space(self, state_space, rng: np.random.RandomState, action_params: Dict):
        control_space = oc.CompoundControlSpace(state_space)

        left_gripper_control_space = oc.RealVectorControlSpace(state_space, 3)
        left_gripper_control_bounds = ob.RealVectorBounds(3)
        # Pitch
        left_gripper_control_bounds.setLow(0, -np.pi)
        left_gripper_control_bounds.setHigh(0, np.pi)
        # Yaw
        left_gripper_control_bounds.setLow(1, -np.pi)
        left_gripper_control_bounds.setHigh(1, np.pi)
        # Displacement
        max_d = action_params['max_distance_gripper_can_move']
        left_gripper_control_bounds.setLow(2, 0)
        left_gripper_control_bounds.setHigh(2, max_d)
        left_gripper_control_space.setBounds(left_gripper_control_bounds)
        control_space.addSubspace(left_gripper_control_space)

        right_gripper_control_space = oc.RealVectorControlSpace(state_space, 3)
        right_gripper_control_bounds = ob.RealVectorBounds(3)
        # Pitch
        right_gripper_control_bounds.setLow(0, -np.pi)
        right_gripper_control_bounds.setHigh(0, np.pi)
        # Yaw
        right_gripper_control_bounds.setLow(1, -np.pi)
        right_gripper_control_bounds.setHigh(1, np.pi)
        # Displacement
        max_d = action_params['max_distance_gripper_can_move']
        right_gripper_control_bounds.setLow(2, 0)
        right_gripper_control_bounds.setHigh(2, max_d)

        right_gripper_control_space.setBounds(right_gripper_control_bounds)
        control_space.addSubspace(right_gripper_control_space)

        def _allocator(cs):
            return DualGripperControlSampler(cs, scenario_ompl=self, rng=rng, action_params=action_params)

        # I override the sampler here so I can use numpy RNG to make things more deterministic.
        # ompl does not allow resetting of seeds, which causes problems when evaluating multiple
        # planning queries in a row.
        control_space.setControlSamplerAllocator(oc.ControlSamplerAllocator(_allocator))

        return control_space

    def make_directed_control_sampler(self,
                                      si: oc.SpaceInformation,
                                      rng: np.random.RandomState,
                                      action_params: Dict,
                                      opt: TrajectoryOptimizer,
                                      max_steps: int):
        return DualGripperDirectedControlSampler(si=si,
                                                 scenario_ompl=self,
                                                 rng=rng,
                                                 action_params=action_params,
                                                 opt=opt,
                                                 max_steps=max_steps)


# noinspection PyMethodOverriding
class DualGripperDirectedControlSampler(oc.DirectedControlSampler):
    def __init__(self,
                 si: oc.SpaceInformation,
                 scenario_ompl: ScenarioOmpl,
                 rng: np.random.RandomState,
                 opt: TrajectoryOptimizer,
                 action_params: Dict,
                 max_steps: int = 50):
        super().__init__(si)
        self.scenario_ompl = scenario_ompl
        self.rng = rng
        self.si = si
        self.action_params = action_params
        self.opt = opt
        self.max_steps = max_steps

    def sampleTo(self, control_out: oc.CompoundControl, _, source: ob.CompoundState, dest: ob.CompoundState):
        current_state_np = self.scenario_ompl.ompl_state_to_numpy(source)
        goal_state_np = self.scenario_ompl.ompl_state_to_numpy(dest)
        initial_actions = [{
            'left_gripper_position':  current_state_np['left_gripper'],
            'right_gripper_position': current_state_np['right_gripper'],
        }]
        control_out_tf, path = self.opt.optimize(environment=self.opt.env,  # FIXME: horrible hack
                                                 goal_state=goal_state_np,
                                                 initial_actions=initial_actions,
                                                 start_state=current_state_np)
        control_out_np = numpify(control_out_tf[0])
        next_state_np = numpify(path[0])
        # FIXME: this is wrong, we actually need to "set" these the same way we do in propagate... right?
        next_state_np['stdev'] = current_state_np['stdev']
        next_state_np['num_diverged'] = current_state_np['num_diverged']
        self.scenario_ompl.numpy_to_ompl_control(current_state_np, control_out_np, control_out)
        self.scenario_ompl.numpy_to_ompl_state(next_state_np, dest)

        step_count = self.rng.randint(1, self.max_steps)
        return step_count


# noinspection PyMethodOverriding
class DualGripperControlSampler(oc.ControlSampler):
    def __init__(self,
                 control_space: oc.CompoundControlSpace,
                 scenario_ompl: ScenarioOmpl,
                 rng: np.random.RandomState,
                 action_params: Dict):
        super().__init__(control_space)
        self.scenario_ompl = scenario_ompl
        self.rng = rng
        self.control_space = control_space
        self.action_params = action_params

    def sampleNext(self, control_out, previous_control, state):
        del previous_control
        del state

        left_phi = self.rng.uniform(-np.pi, np.pi)
        right_phi = self.rng.uniform(-np.pi, np.pi)
        left_theta = self.rng.uniform(-np.pi, np.pi)
        right_theta = self.rng.uniform(-np.pi, np.pi)
        left_r = self.rng.uniform(0, self.action_params['max_distance_gripper_can_move'])
        right_r = self.rng.uniform(0, self.action_params['max_distance_gripper_can_move'])

        control_out[0][0] = left_r
        control_out[0][1] = left_phi
        control_out[0][2] = left_theta

        control_out[1][0] = right_r
        control_out[1][1] = right_phi
        control_out[1][2] = right_theta

    def sampleStepCount(self, min_steps, max_steps):
        step_count = self.rng.randint(min_steps, max_steps)
        return step_count


# noinspection PyMethodOverriding
class DualGripperStateSampler(ob.CompoundStateSampler):

    def __init__(self,
                 state_space,
                 scenario_ompl: FloatingRopeOmpl,
                 extent,
                 rng: np.random.RandomState,
                 plot: bool):
        super().__init__(state_space)
        self.state_space = state_space
        self.scenario_ompl = scenario_ompl
        self.extent = np.array(extent).reshape(3, 2)
        self.rng = rng
        self.plot = plot

        bbox_msg = extent_to_bbox(extent)
        bbox_msg.header.frame_id = 'world'
        self.sampler_extents_bbox_pub = rospy.Publisher('sampler_extents', BoundingBox, queue_size=10, latch=True)
        self.sampler_extents_bbox_pub.publish(bbox_msg)

    def sampleUniform(self, state_out: ob.CompoundState):
        random_point = self.rng.uniform(self.extent[:, 0], self.extent[:, 1])
        random_point_rope = np.concatenate([random_point] * FloatingRopeScenario.n_links)
        state_np = {
            'left_gripper':  random_point,
            'right_gripper': random_point,
            'rope':          random_point_rope,
            'num_diverged':  np.zeros(1, dtype=np.float64),
            'stdev':         np.zeros(1, dtype=np.float64),
        }
        self.scenario_ompl.numpy_to_ompl_state(state_np, state_out)

        if self.plot:
            self.scenario_ompl.s.plot_sampled_state(state_np)


# noinspection PyMethodOverriding
class DualGripperGoalRegion(ob.GoalSampleableRegion):

    def __init__(self,
                 si: oc.SpaceInformation,
                 scenario_ompl: FloatingRopeOmpl,
                 rng: np.random.RandomState,
                 threshold: float,
                 goal: Dict,
                 plot: bool):
        super(DualGripperGoalRegion, self).__init__(si)
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
        distance = float(self.scenario_ompl.s.distance_to_gripper_goal(state_np, self.goal).numpy())

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
        rope = sample_rope_grippers(self.rng,
                                    self.goal['left_gripper'],
                                    self.goal['right_gripper'],
                                    FloatingRopeScenario.n_links)

        goal_state_np = {
            'left_gripper':  self.goal['left_gripper'],
            'right_gripper': self.goal['right_gripper'],
            'rope':          rope.flatten(),
            'num_diverged':  np.zeros(1, dtype=np.float64),
            'stdev':         np.zeros(1, dtype=np.float64),
        }

        self.scenario_ompl.numpy_to_ompl_state(goal_state_np, state_out)

        if self.plot:
            self.scenario_ompl.s.plot_sampled_goal_state(goal_state_np)

    def maxSampleCount(self):
        return 100


# noinspection PyMethodOverriding
class RopeMidpointGoalRegion(ob.GoalSampleableRegion):

    def __init__(self,
                 si: oc.SpaceInformation,
                 scenario_ompl: FloatingRopeOmpl,
                 rng: np.random.RandomState,
                 threshold: float,
                 goal: Dict,
                 plot: bool):
        super(RopeMidpointGoalRegion, self).__init__(si)
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
        distance = float(self.scenario_ompl.s.distance_to_midpoint_goal(state_np, self.goal).numpy())

        # this ensures the goal must have num_diverged = 0
        if state_np['num_diverged'] > 0:
            distance = 1e9
        return distance

    def sampleGoal(self, state_out: ob.CompoundState):
        d = self.getThreshold()
        random_direction = transformations.random_rotation_matrix(self.rng.uniform(0, 1, [3])) @ np.array([d, 0, 0, 1])
        random_distance = self.rng.uniform(0.0, d)
        random_direction = random_direction[:3]
        print('DIST:', random_distance)
        random_point = self.goal['midpoint'] + random_direction * random_distance

        goal_state_np = {
            'left_gripper':  random_point,
            'right_gripper': random_point,
            'rope':          [random_point] * self.scenario_ompl.s.n_links,
            'num_diverged':  np.zeros(1, dtype=np.float64),
            'stdev':         np.zeros(1, dtype=np.float64),
        }

        self.scenario_ompl.numpy_to_ompl_state(goal_state_np, state_out)

        if self.plot:
            self.scenario_ompl.s.plot_sampled_goal_state(goal_state_np)

    def maxSampleCount(self):
        return 100


# noinspection PyMethodOverriding
class RopeAnyPointGoalRegion(ob.GoalSampleableRegion):

    def __init__(self,
                 si: oc.SpaceInformation,
                 scenario_ompl: FloatingRopeOmpl,
                 rng: np.random.RandomState,
                 threshold: float,
                 goal: Dict,
                 plot: bool):
        super(RopeAnyPointGoalRegion, self).__init__(si)
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
        distance = float(self.scenario_ompl.s.distance_to_any_point_goal(state_np, self.goal).numpy())

        # this ensures the goal must have num_diverged = 0
        if state_np['num_diverged'] > 0:
            distance = 1e9
        return distance

    def sampleGoal(self, state_out: ob.CompoundState):
        d = self.getThreshold()
        random_distance = self.rng.uniform(0.0, d)
        random_direction = transformations.random_rotation_matrix(self.rng.uniform(0, 1, [3])) * np.array([d, 0, 0, 1])
        random_point = self.goal['point'] + random_direction * random_distance

        goal_state_np = {
            'left_gripper':  random_point,
            'right_gripper': random_point,
            'rope':          [random_point] * self.scenario_ompl.s.n_links,
            'num_diverged':  np.zeros(1, dtype=np.float64),
            'stdev':         np.zeros(1, dtype=np.float64),
        }

        self.scenario_ompl.numpy_to_ompl_state(goal_state_np, state_out)

        if self.plot:
            self.scenario_ompl.s.plot_sampled_goal_state(goal_state_np)

    def maxSampleCount(self):
        return 100


# noinspection PyMethodOverriding
class RopeAndGrippersGoalRegion(ob.GoalSampleableRegion):

    def __init__(self,
                 si: oc.SpaceInformation,
                 scenario_ompl: FloatingRopeOmpl,
                 rng: np.random.RandomState,
                 threshold: float,
                 goal: Dict,
                 plot: bool):
        super(RopeAndGrippersGoalRegion, self).__init__(si)
        self.setThreshold(threshold)
        self.goal = goal
        self.scenario_ompl = scenario_ompl
        self.rng = rng
        self.plot = plot

    def distanceGoal(self, state: ob.CompoundState):
        state_np = self.scenario_ompl.ompl_state_to_numpy(state)
        distance = float(self.scenario_ompl.s.distance_grippers_and_any_point_goal(state_np, self.goal).numpy())

        # this ensures the goal must have num_diverged = 0
        if state_np['num_diverged'] > 0:
            distance = 1e9
        return distance

    def sampleGoal(self, state_out: ob.CompoundState):
        # attempt to sample "legit" rope states
        kd = 0.05
        rope = sample_rope_and_grippers(
            self.rng, self.goal['left_gripper'], self.goal['right_gripper'], self.goal['point'],
            FloatingRopeScenario.n_links,
            kd)

        goal_state_np = {
            'left_gripper':  self.goal['left_gripper'],
            'right_gripper': self.goal['right_gripper'],
            'rope':          rope.flatten(),
            'num_diverged':  np.zeros(1, dtype=np.float64),
            'stdev':         np.zeros(1, dtype=np.float64),
        }

        self.scenario_ompl.numpy_to_ompl_state(goal_state_np, state_out)

        if self.plot:
            self.scenario_ompl.s.plot_sampled_goal_state(goal_state_np)

    def maxSampleCount(self):
        return 100


# noinspection PyMethodOverriding
class RopeAndGrippersBoxesGoalRegion(ob.GoalSampleableRegion):

    def __init__(self,
                 si: oc.SpaceInformation,
                 scenario_ompl: FloatingRopeOmpl,
                 rng: np.random.RandomState,
                 threshold: float,
                 goal: Dict,
                 plot: bool):
        super(RopeAndGrippersBoxesGoalRegion, self).__init__(si)
        self.goal = goal
        self.scenario_ompl = scenario_ompl
        self.setThreshold(threshold)
        self.rng = rng
        self.plot = plot

    def isSatisfied(self, state: ob.CompoundState, distance):
        state_np = self.scenario_ompl.ompl_state_to_numpy(state)
        rope_points = np.reshape(state_np['rope'], [-1, 3])
        n_from_ends = 7
        near_center_rope_points = rope_points[n_from_ends:-n_from_ends]

        left_gripper_extent = np.reshape(self.goal['left_gripper_box'], [3, 2])
        left_gripper_satisfied = np.logical_and(
            state_np['left_gripper'] >= left_gripper_extent[:, 0],
            state_np['left_gripper'] <= left_gripper_extent[:, 1])

        right_gripper_extent = np.reshape(self.goal['right_gripper_box'], [3, 2])
        right_gripper_satisfied = np.logical_and(
            state_np['right_gripper'] >= right_gripper_extent[:, 0],
            state_np['right_gripper'] <= right_gripper_extent[:, 1])

        point_extent = np.reshape(self.goal['point_box'], [3, 2])
        points_satisfied = np.logical_and(near_center_rope_points >=
                                          point_extent[:, 0], near_center_rope_points <= point_extent[:, 1])
        any_point_satisfied = np.reduce_any(points_satisfied)

        return float(any_point_satisfied and left_gripper_satisfied and right_gripper_satisfied)

    def sampleGoal(self, state_out: ob.CompoundState):
        # attempt to sample "legit" rope states
        kd = 0.05
        rope = sample_rope_and_grippers(
            self.rng, self.goal['left_gripper'], self.goal['right_gripper'], self.goal['point'],
            FloatingRopeScenario.n_links,
            kd)

        goal_state_np = {
            'left_gripper':  self.goal['left_gripper'],
            'right_gripper': self.goal['right_gripper'],
            'rope':          rope.flatten(),
            'num_diverged':  np.zeros(1, dtype=np.float64),
            'stdev':         np.zeros(1, dtype=np.float64),
        }

        self.scenario_ompl.numpy_to_ompl_state(goal_state_np, state_out)

        if self.plot:
            self.scenario_ompl.s.plot_sampled_goal_state(goal_state_np)

    def distanceGoal(self, state: ob.CompoundState):
        state_np = self.scenario_ompl.ompl_state_to_numpy(state)
        distance = float(self.scenario_ompl.s.distance_grippers_and_any_point_goal(state_np, self.goal).numpy())

        # this ensures the goal must have num_diverged = 0
        if state_np['num_diverged'] > 0:
            distance = 1e9
        return distance

    def maxSampleCount(self):
        return 100
