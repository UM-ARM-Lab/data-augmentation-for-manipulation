from dm_control import mjcf, composer
from dm_control.composer import Observables
from dm_control.composer.observation import observable
from dm_control.entities.manipulators.base import RobotHand


class PrimitiveHandObservables(Observables):

    @composer.observable
    def tcp_pos(self):
        return observable.MJCFFeature('xpos', self._entity.tool_center_point)

    @composer.observable
    def tcp_xmat(self):
        return observable.MJCFFeature('xmat', self._entity.tool_center_point)


HEIGHT = 0.08
HALF_HEIGHT = HEIGHT / 2
RADIUS = 0.02
Z_OFFSET = HEIGHT


class PrimitiveHand(RobotHand):

    def _build(self):
        self._mjcf_root = mjcf.RootElement("primitive_hand")

        self.thigh = self.mjcf_model.worldbody.add('body')
        self.thigh.add('geom',
                       type='cylinder',
                       size=[RADIUS, HALF_HEIGHT],
                       rgba=[0, 1, 0, 1],
                       pos=[0, 0, HALF_HEIGHT])

        self._bodies = self.mjcf_model.find_all('body')
        self._tool_center_point = self.mjcf_model.worldbody.add('site', name='tcp', pos=[0, 0, 0], euler=[0, 0, 0])

    def _build_observables(self):
        return PrimitiveHandObservables(self)

    @property
    def tool_center_point(self):
        return self._tool_center_point

    @property
    def mjcf_model(self):
        return self._mjcf_root

    @property
    def actuators(self):
        """List of finger actuators."""
        return []

    def set_grasp(self, physics, close_factors):
        pass
