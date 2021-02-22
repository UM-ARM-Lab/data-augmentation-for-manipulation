import warnings
from typing import Dict

import numpy as np

from arc_utilities.transformation_helper import vector3_to_spherical, spherical_to_vector3
from link_bot_planning import floating_rope_ompl
from link_bot_planning.floating_rope_ompl import FloatingRopeOmpl, DualGripperControlSampler, sample_rope_and_grippers, \
    make_random_rope_and_grippers_for_goal_point, sample_rope_and_grippers_from_extent
from link_bot_planning.my_planner import SharedPlanningStateOMPL
from link_bot_pycommon.floating_rope_scenario import FloatingRopeScenario

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    import ompl.base as ob
    import ompl.control as oc

import rospy
from jsk_recognition_msgs.msg import BoundingBox
from link_bot_pycommon.bbox_visualization import extent_to_bbox


class DualArmRopeOmpl(FloatingRopeOmpl):

    def numpy_to_ompl_state(self, state_np: Dict, state_out: ob.CompoundState):
        rope_points = np.reshape(state_np['rope'], [-1, 3])

        state_component_idx = 0
        for i in range(3):
            state_out[state_component_idx][i] = np.float64(state_np['left_gripper'][i])
        state_component_idx += 1

        for i in range(3):
            state_out[state_component_idx][i] = np.float64(state_np['right_gripper'][i])
        state_component_idx += 1

        for j in range(FloatingRopeScenario.n_links):
            for i in range(3):
                state_out[state_component_idx][i] = np.float64(rope_points[j][i])
            state_component_idx += 1

        state_out[state_component_idx][0] = np.float64(state_np['stdev'][0])
        state_component_idx += 1

        state_out[state_component_idx][0] = np.float64(state_np['num_diverged'][0])
        state_component_idx += 1

        joint_positions = state_np['joint_positions']
        for i, joint_state_i in enumerate(joint_positions):
            state_out[state_component_idx][i] = np.float64(joint_state_i)
        state_component_idx += 1

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
        state_component_idx = 0
        left_gripper = np.array([ompl_state[state_component_idx][0],
                                 ompl_state[state_component_idx][1],
                                 ompl_state[state_component_idx][2]], np.float32)
        state_component_idx += 1

        right_gripper = np.array([ompl_state[state_component_idx][0],
                                  ompl_state[state_component_idx][1],
                                  ompl_state[state_component_idx][2]], np.float32)
        state_component_idx += 1

        rope = []
        for i in range(FloatingRopeScenario.n_links):
            rope.append(ompl_state[state_component_idx][0])
            rope.append(ompl_state[state_component_idx][1])
            rope.append(ompl_state[state_component_idx][2])
            state_component_idx += 1
        rope = np.array(rope, np.float32)

        stdev = np.array([ompl_state[state_component_idx][0]], np.float32)
        state_component_idx += 1

        num_diverged = np.array([ompl_state[state_component_idx][0]], np.float32)
        state_component_idx += 1

        joint_positions_subspace = self.state_space.getSubspace("joint_positions")
        n_joints = joint_positions_subspace.getDimension()
        joint_positions = []
        for i in range(n_joints):
            joint_positions.append(ompl_state[state_component_idx][i])
        joint_positions = np.array(joint_positions, np.float32)

        joint_names = [joint_positions_subspace.getDimensionName(i) for i in range(n_joints)]

        return {
            'left_gripper':    left_gripper,
            'right_gripper':   right_gripper,
            'rope':            rope,
            'stdev':           stdev,
            'num_diverged':    num_diverged,
            'joint_positions': joint_positions,
            'joint_names':     joint_names,
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

    def make_state_space(self):
        state_space = ob.CompoundStateSpace()

        min_x, max_x, min_y, max_y, min_z, max_z = self.planner_params['extent']

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

        # joint state
        joint_names = self.s.get_joint_names()
        n_joints = len(joint_names)
        joint_positions_subspace = ob.RealVectorStateSpace(n_joints)
        joint_positions_bounds = ob.RealVectorBounds(n_joints)
        # no need to set joint limits here because we will be checking these when we call follow_jacobian
        joint_positions_bounds.setLow(-1000)
        joint_positions_bounds.setHigh(1000)
        joint_positions_subspace.setBounds(joint_positions_bounds)
        joint_positions_subspace.setName("joint_positions")
        for i, joint_name in enumerate(joint_names):
            joint_positions_subspace.setDimensionName(i, joint_name)
        # by setting weight to zero we make is unnecessary to sample joint states
        state_space.addSubspace(joint_positions_subspace, weight=0)

        def _state_sampler_allocator(state_space):
            return DualGripperStateSampler(state_space,
                                           scenario_ompl=self,
                                           extent=self.planner_params['state_sampler_extent'],
                                           rng=self.state_sampler_rng,
                                           plot=self.plot)

        state_space.setStateSamplerAllocator(ob.StateSamplerAllocator(_state_sampler_allocator))

        return state_space

    def make_control_space(self):
        control_space = oc.CompoundControlSpace(self.state_space)

        left_gripper_control_space = oc.RealVectorControlSpace(self.state_space, 3)
        left_gripper_control_bounds = ob.RealVectorBounds(3)
        # Pitch
        left_gripper_control_bounds.setLow(0, -np.pi)
        left_gripper_control_bounds.setHigh(0, np.pi)
        # Yaw
        left_gripper_control_bounds.setLow(1, -np.pi)
        left_gripper_control_bounds.setHigh(1, np.pi)
        # Displacement
        max_d = self.action_params['max_distance_gripper_can_move']
        left_gripper_control_bounds.setLow(2, 0)
        left_gripper_control_bounds.setHigh(2, max_d)
        left_gripper_control_space.setBounds(left_gripper_control_bounds)
        control_space.addSubspace(left_gripper_control_space)

        right_gripper_control_space = oc.RealVectorControlSpace(self.state_space, 3)
        right_gripper_control_bounds = ob.RealVectorBounds(3)
        # Pitch
        right_gripper_control_bounds.setLow(0, -np.pi)
        right_gripper_control_bounds.setHigh(0, np.pi)
        # Yaw
        right_gripper_control_bounds.setLow(1, -np.pi)
        right_gripper_control_bounds.setHigh(1, np.pi)
        # Displacement
        max_d = self.action_params['max_distance_gripper_can_move']
        right_gripper_control_bounds.setLow(2, 0)
        right_gripper_control_bounds.setHigh(2, max_d)

        right_gripper_control_space.setBounds(right_gripper_control_bounds)
        control_space.addSubspace(right_gripper_control_space)

        def _allocator(cs):
            return DualGripperControlSampler(cs,
                                             scenario_ompl=self,
                                             rng=self.control_sampler_rng,
                                             shared_planning_state=self.sps,
                                             action_params=self.action_params)

        # I override the sampler here so I can use numpy RNG to make things more deterministic.
        # ompl does not allow resetting of seeds, which causes problems when evaluating multiple
        # planning queries in a row.
        control_space.setControlSamplerAllocator(oc.ControlSamplerAllocator(_allocator))

        return control_space

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
                                          shared_planning_state=self.sps,
                                          plot=plot)
        else:
            raise NotImplementedError()


# noinspection PyMethodOverriding
class DualGripperStateSampler(floating_rope_ompl.DualGripperStateSampler):

    def sampleUniform(self, state_out: ob.CompoundState):
        left_gripper, random_rope, right_gripper = sample_rope_and_grippers_from_extent(self.rng, self.extent)

        n_joints = self.scenario_ompl.state_space.getSubspace("joint_positions").getDimension()
        joint_positions = np.zeros(n_joints, dtype=np.float64)
        state_np = {
            'left_gripper':    left_gripper,
            'right_gripper':   right_gripper,
            'rope':            random_rope,
            'num_diverged':    np.zeros(1, dtype=np.float64),
            'stdev':           np.zeros(1, dtype=np.float64),
            'joint_positions': joint_positions,
        }
        self.scenario_ompl.numpy_to_ompl_state(state_np, state_out)

        if self.plot:
            self.scenario_ompl.s.plot_sampled_state(state_np)


# noinspection PyMethodOverriding
class RopeMidpointGoalRegion(floating_rope_ompl.RopeMidpointGoalRegion):

    def __init__(self,
                 si: oc.SpaceInformation,
                 scenario_ompl: FloatingRopeOmpl,
                 rng: np.random.RandomState,
                 threshold: float,
                 goal: Dict,
                 plot: bool):
        super().__init__(si, scenario_ompl, rng, threshold, goal, plot)
        self.n_joints = self.scenario_ompl.state_space.getSubspace("joint_positions").getDimension()

    def make_goal_state(self, random_point):
        left_gripper, random_rope, right_gripper = make_random_rope_and_grippers_for_goal_point(self.rng, random_point)

        goal_state_np = {
            'left_gripper':    left_gripper,
            'right_gripper':   right_gripper,
            'rope':            random_rope,
            'num_diverged':    np.zeros(1, dtype=np.float64),
            'stdev':           np.zeros(1, dtype=np.float64),
            'joint_positions': np.zeros(self.n_joints, dtype=np.float64),
        }
        return goal_state_np


# noinspection PyMethodOverriding
class RopeAnyPointGoalRegion(floating_rope_ompl.RopeAnyPointGoalRegion):

    def __init__(self,
                 si: oc.SpaceInformation,
                 scenario_ompl: FloatingRopeOmpl,
                 rng: np.random.RandomState,
                 threshold: float,
                 goal: Dict,
                 shared_planning_state: SharedPlanningStateOMPL,
                 plot: bool,
                 ):
        super().__init__(si, scenario_ompl, rng, threshold, goal, shared_planning_state, plot)
        self.n_joints = self.scenario_ompl.state_space.getSubspace("joint_positions").getDimension()

    def make_goal_state(self, random_point: np.array):
        left_gripper, random_rope, right_gripper = make_random_rope_and_grippers_for_goal_point(self.rng, random_point)

        goal_state_np = {
            'left_gripper':    left_gripper,
            'right_gripper':   right_gripper,
            'rope':            random_rope,
            'num_diverged':    np.zeros(1, dtype=np.float64),
            'stdev':           np.zeros(1, dtype=np.float64),
            'joint_positions': np.zeros(self.n_joints, dtype=np.float64),
        }
        return goal_state_np
