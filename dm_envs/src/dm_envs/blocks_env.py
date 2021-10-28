from typing import Dict

from dm_control import composer
from dm_control.composer import initializers
from dm_control.composer.observation import observable
from dm_control.composer.variation import distributions
from dm_control.entities.manipulators.base import DOWN_QUATERNION
from dm_control.manipulation.props.primitive import Box
from dm_control.manipulation.shared import arenas, cameras, workspaces, constants, robots
from dm_control.manipulation.shared import registry, tags
from dm_control.manipulation.shared.observations import VISION
from dm_control.utils import inverse_kinematics

from dm_envs import primitive_hand
from dm_envs.primitive_hand import PrimitiveHand


class MyBlocks(composer.Task):
    def __init__(self, arena, arm, hand, obs_settings, control_timestep, params):
        self.box_length = 0.02
        self._arena = arena
        self._arm = arm
        self._hand = hand
        self._arm.attach(self._hand)
        self._arena.attach_offset(self._arm, offset=robots.ARM_OFFSET)
        self.control_timestep = control_timestep

        # extents are ordered [xmin, xmax, ymin, ymax, zmin, zmax
        self._prop_bbox = workspaces.BoundingBox(lower=params['extent'][0::2],
                                                 upper=params['extent'][1::2])
        self._tcp_bbox = workspaces.BoundingBox(lower=params['gripper_action_sample_extent'][0::2],
                                                upper=params['gripper_action_sample_extent'][1::2])

        # Create initializers
        self._block_placer = None

        start_pos = [params['gripper_action_sample_extent'][0],
                     params['gripper_action_sample_extent'][2],
                     primitive_hand.Z_OFFSET + 0.01]  # start 1cm off the floor/table
        self._tcp_initializer = initializers.ToolCenterPointInitializer(self._hand,
                                                                        self._arm,
                                                                        position=start_pos,
                                                                        quaternion=workspaces.DOWN_QUATERNION)

        # configure physics
        self._arena.mjcf_model.size.nconmax = 10000
        self._arena.mjcf_model.size.njmax = 10000

        # create block entities
        self._blocks = []
        for i in range(params['num_blocks']):
            block = Box(half_lengths=[self.box_length / 2] * 3, name=f'box{i}')
            self._arena.add_free_entity(block)
            self._blocks.append(block)

        # configure and enable observables
        self._task_observables = cameras.add_camera_observables(arena, obs_settings, cameras.FRONT_CLOSE)

        def _num_blocks_observable_callable(_):
            return params['num_blocks']

        self._task_observables['num_blocks'] = observable.Generic(_num_blocks_observable_callable)
        self._task_observables['num_blocks'].enabled = True
        self._hand.observables.enable_all()

        for block in self._blocks:
            block.observables.position.enabled = True
            block.observables.orientation.enabled = True

    @property
    def root_entity(self):
        return self._arena

    @property
    def task_observables(self):
        return self._task_observables

    def initialize_episode_mjcf(self, random_state):
        # We need to define the prop initializer for the blocks here rather than in
        # the `__init__`, since `PropPlacer` looks for freejoints on instantiation.
        self._block_placer = initializers.PropPlacer(
            props=self._blocks,
            position=distributions.Uniform(*self._prop_bbox),
            quaternion=workspaces.uniform_z_rotation,
            settle_physics=True)

    def initialize_episode(self, physics, random_state):
        self._tcp_initializer(physics, random_state)
        self._block_placer(physics, random_state)

    def get_reward(self, physics):
        return 0

    @property
    def joint_names(self):
        joint_names = [joint.full_identifier for joint in self._arm.joints]
        return joint_names

    def solve_position_ik(self, physics, target_pos):
        success = False
        for _ in range(10):
            result = inverse_kinematics.qpos_from_site_pose(
                physics=physics,
                site_name='jaco_arm/primitive_hand/tcp',
                target_pos=target_pos,
                target_quat=DOWN_QUATERNION,
                joint_names=self.joint_names,
                rot_weight=2)

            if result.success:
                success = True
                break

        indices = [physics.model.name2id(joint_name, 'joint') for joint_name in self.joint_names]
        joint_position = result.qpos[indices]

        return success, joint_position


@registry.add(tags.VISION)
def my_blocks(params: Dict):
    arena = arenas.Standard()
    arm = robots.make_arm(obs_settings=VISION)
    hand = PrimitiveHand()
    task = MyBlocks(arena, arm, hand, VISION, constants.CONTROL_TIMESTEP, params)
    return task


def register_envs():
    pass
