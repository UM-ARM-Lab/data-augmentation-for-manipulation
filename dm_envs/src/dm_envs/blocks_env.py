import collections

from dm_control import composer
from dm_control.composer import initializers
from dm_control.composer.variation import distributions
from dm_control.manipulation.props.primitive import Box
from dm_control.manipulation.shared import arenas, cameras, workspaces, constants, robots
from dm_control.manipulation.shared import registry, tags
from dm_control.manipulation.shared.observations import ObservationSettings, _ENABLED_FEATURE, _ENABLED_FTT, \
    ObservableSpec, VISION


class MyCameraObservableSpec(collections.namedtuple(
    'CameraObservableSpec', ('depth', 'height', 'width') + ObservableSpec._fields)):
    __slots__ = ()


ENABLED_CAMERA_DEPTH = MyCameraObservableSpec(height=84,
                                              width=84,
                                              enabled=True,
                                              depth=True,
                                              update_interval=1,
                                              buffer_size=1,
                                              delay=0,
                                              aggregator=None,
                                              corruptor=None)

VISION_DEPTH = ObservationSettings(proprio=_ENABLED_FEATURE,
                                   ftt=_ENABLED_FTT,
                                   prop_pose=_ENABLED_FEATURE,
                                   camera=ENABLED_CAMERA_DEPTH)

MyWorkspace = collections.namedtuple('MyWorkspace', ['prop_bbox', 'tcp_bbox', 'arm_offset'])

WORKSPACE_PROP_W = 0.15
MY_WORKSPACE = MyWorkspace(prop_bbox=workspaces.BoundingBox(lower=(-WORKSPACE_PROP_W, -WORKSPACE_PROP_W, 1e-6),
                                                            upper=(WORKSPACE_PROP_W, WORKSPACE_PROP_W, 0.10)),
                           tcp_bbox=workspaces.BoundingBox(lower=(-0.1, -0.1, 0.10), upper=(0.1, 0.1, 0.1)),
                           arm_offset=robots.ARM_OFFSET)


class MyBlocks(composer.Task):
    def __init__(self, arena, arm, hand, obs_settings, workspace, control_timestep, num_blocks: int):
        self.box_length = 0.02
        self._arena = arena
        self._arm = arm
        self._hand = hand
        self._arm.attach(self._hand)
        self._arena.attach_offset(self._arm, offset=workspace.arm_offset)
        self.control_timestep = control_timestep
        self._prop_bbox = workspace.prop_bbox

        # Create initializers
        self._block_placer = None
        self._tcp_initializer = initializers.ToolCenterPointInitializer(
            self._hand, self._arm,
            position=distributions.Uniform(*workspace.tcp_bbox),
            quaternion=workspaces.DOWN_QUATERNION)

        # configure physics
        self._arena.mjcf_model.size.nconmax = 10000
        self._arena.mjcf_model.size.njmax = 10000

        # create block entities
        self._blocks = []
        for i in range(num_blocks):
            block = Box(half_lengths=[self.box_length / 2] * 3)
            self._arena.add_free_entity(block)
            self._blocks.append(block)

        # configure and enable observables
        self._task_observables = cameras.add_camera_observables(arena, obs_settings, cameras.FRONT_CLOSE)
        for block in self._blocks:
            block.observables.position.enabled = True
            block.observables.orientation.enabled = True

    @property
    def root_entity(self):
        return self._arena

    @property
    def arm(self):
        return self._arm

    @property
    def hand(self):
        return self._hand

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
        self._hand.set_grasp(physics, close_factors=random_state.uniform())
        self._tcp_initializer(physics, random_state)
        self._block_placer(physics, random_state)

    def get_reward(self, physics):
        return 0


@registry.add(tags.VISION)
def my_blocks(num_blocks=10):
    arena = arenas.Standard()
    arm = robots.make_arm(obs_settings=VISION)
    hand = robots.make_hand(obs_settings=VISION)
    task = MyBlocks(arena, arm, hand, VISION, MY_WORKSPACE, constants.CONTROL_TIMESTEP, num_blocks=num_blocks)
    return task


def register_envs():
    pass
