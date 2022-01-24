from dm_control.composer import define
from dm_control.composer.observation import observable
from dm_control.entities.manipulators.base import RobotHand
from dm_control.entities.props import Primitive
from dm_control.entities.props.primitive import PrimitiveObservables

HEIGHT = 0.08
HALF_HEIGHT = HEIGHT / 2
RADIUS = 0.02


class PrimitiveHandObservables(PrimitiveObservables):

    @define.observable
    def tcp_pos(self):
        return observable.MJCFFeature('xpos', self._entity.tool_center_point)


class PrimitiveHand(RobotHand, Primitive):

    def _build(self, hand_name):
        size = [RADIUS, HALF_HEIGHT]
        Primitive._build(self,
                         geom_type='cylinder',
                         size=size,
                         name=hand_name,
                         pos=[0, 0, HALF_HEIGHT],
                         rgba=[1, 0, 1, 1])
        self._tool_center_point = self.mjcf_model.worldbody.add('site', name='tcp', pos=[0, 0, HEIGHT], euler=[0, 0, 0])

    def _build_observables(self):
        return PrimitiveHandObservables(self)

    @property
    def tool_center_point(self):
        return self._tool_center_point

    @property
    def actuators(self):
        return []

    def set_grasp(self, physics, close_factors):
        raise NotImplementedError()
