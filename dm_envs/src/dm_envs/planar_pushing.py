from typing import Dict

from dm_control import composer
from dm_control.composer import initializers
from dm_control.composer.observation import observable
from dm_control.composer.variation import distributions
from dm_control.entities.manipulators.base import DOWN_QUATERNION
from dm_control.manipulation.shared import cameras, workspaces, robots, observations, constants, arenas
from dm_control.utils import inverse_kinematics

from dm_envs import primitive_hand
from dm_envs.primitive_hand import PrimitiveHand


class PlanarPushingTask(composer.Task):
    def __init__(self,
                 params: Dict,
                 arena=arenas.Standard(),
                 arm=None,
                 hand=PrimitiveHand(),
                 obs_settings=observations.VISION,
                 control_timestep=constants.CONTROL_TIMESTEP,
                 ):

        self._arena = arena

        if arm is None:
            arm = robots.make_arm(obs_settings)
        self._arm = arm
        self._hand = hand
        self._arm.attach(self._hand)
        self._arena.attach_offset(self._arm, offset=robots.ARM_OFFSET)
        self.control_timestep = control_timestep

        # extents are ordered [xmin, xmax, ymin, ymax, zmin, zmax
        self._prop_bbox = workspaces.BoundingBox(lower=params['objs_init_extent'][0::2],
                                                 upper=params['objs_init_extent'][1::2])
        self._tcp_bbox = workspaces.BoundingBox(lower=params['gripper_action_sample_extent'][0::2],
                                                upper=params['gripper_action_sample_extent'][1::2])

        # Create initializers
        self._obj_placer = None

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

        self.objs = self.create_objs(params)

        # configure and enable observables
        self._task_observables = cameras.add_camera_observables(arena, obs_settings, cameras.FRONT_CLOSE)

        def _dt_observable_callable(_):
            return control_timestep

        self._task_observables['dt'] = observable.Generic(_dt_observable_callable)
        self._task_observables['dt'].enabled = True

        self._hand.observables.enable_all()
        for obj_observable_name, obj_observable in self.create_objs_observables(params).items():
            self._task_observables[obj_observable_name] = obj_observable
            self._task_observables[obj_observable_name].enabled = True

        for obj in self.objs:
            obj.observables.enable_all()

    def create_objs(self, params):
        raise NotImplementedError()

    def create_objs_observables(self, params):
        raise NotImplementedError()

    @property
    def root_entity(self):
        return self._arena

    @property
    def task_observables(self):
        return self._task_observables

    def initialize_episode_mjcf(self, random_state):
        # We need to define the prop initializer for the blocks here rather than in
        # the `__init__`, since `PropPlacer` looks for freejoints on instantiation.
        self._obj_placer = initializers.PropPlacer(
            props=self.objs,
            position=distributions.Uniform(*self._prop_bbox),
            quaternion=workspaces.uniform_z_rotation,
            settle_physics=True)

    def initialize_episode(self, physics, random_state):
        self._tcp_initializer(physics, random_state)
        self._obj_placer(physics, random_state)

    def get_reward(self, physics):
        return 0

    @property
    def joint_names(self):
        joint_names = [joint.full_identifier for joint in self._arm.joints]
        return joint_names

    def solve_position_ik(self, physics, target_pos):
        initial_qpos = physics.bind(self._arm.joints).qpos.copy()

        result = inverse_kinematics.qpos_from_site_pose(
            physics=physics,
            site_name='jaco_arm/primitive_hand/tcp',
            target_pos=target_pos,
            target_quat=DOWN_QUATERNION,
            joint_names=self.joint_names,
            rot_weight=2,  # more rotation weight than the default
            inplace=True,
        )

        joint_position = physics.named.data.qpos[self.joint_names]

        # reset the arm joints to their original positions, because the above functions actually modify physics state
        physics.bind(self._arm.joints).qpos = initial_qpos

        return result.success, joint_position
