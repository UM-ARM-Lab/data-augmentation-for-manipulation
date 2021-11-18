import numpy as np

from dm_envs import primitive_hand
from dm_envs.cylinders_task import PlanarPushingCylindersTask


def push(angle, init_pos, distance, z):
    """

    Args:
        angle: in radians, with respect to the x axis
        init_pos: [x,y] or [x,y,z]
        distance: scalar
        z: scalar

    Returns: the new [x,y,z]

    """
    sin = np.sin(angle)
    cos = np.cos(angle)
    x = init_pos[0] + distance * cos
    y = init_pos[1] + distance * sin
    return np.array([x, y, z])


class ToyExampleTask(PlanarPushingCylindersTask):

    def __init__(self, params):
        super().__init__(params)

        self._arena.mjcf_model.size.nconmax = 100
        self._arena.mjcf_model.size.njmax = 100

    def initialize_episode_mjcf(self, random_state):
        pass

    def initialize_episode(self, physics, random_state):
        self._tcp_initializer(physics, random_state)

        # get the position of the tcp
        tcp_pos = self.observables['jaco_arm/primitive_hand/tcp_pos'](physics)

        angle = self.params['push_angle']
        radius = self.params['radius']
        distance = primitive_hand.RADIUS + radius + 0.001

        target_pos = push(angle, tcp_pos, distance, self.params['height'] / 2 + 0.001)

        assert len(self.objs) == 1
        prop = self.objs[0]
        prop.set_pose(physics, target_pos, [1, 0, 0, 0])

        # let stuff settle
        for _ in range(100):
            physics.step()
