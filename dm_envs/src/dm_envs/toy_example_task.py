from dm_envs import primitive_hand
from dm_envs.cylinders_task import PlanarPushingCylindersTask


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

        # add an offset
        radius = self.params['radius']
        target_x = tcp_pos[0] + primitive_hand.RADIUS + radius + 0.001
        target_y = tcp_pos[1]
        target_z = self.params['height'] / 2 + 0.001

        assert len(self.objs) == 1
        prop = self.objs[0]
        prop.set_pose(physics, [target_x, target_y, target_z], [1, 0, 0, 0])

        # let stuff settle
        for _ in range(100):
            physics.step()
