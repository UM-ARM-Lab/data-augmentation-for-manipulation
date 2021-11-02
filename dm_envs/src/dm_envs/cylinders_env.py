from dm_control.composer.observation import observable
from dm_control.manipulation.props.primitive import Cylinder

from dm_envs.planar_pushing import PlanarPushingTask


class PlanarPushingCylindersTask(PlanarPushingTask):
    def create_objs(self, params):
        _objs = []
        for i in range(params['num_objs']):
            obj = Cylinder(radius=params['radius'], half_length=params['height'] / 2, name=f'obj{i}')
            self._arena.add_free_entity(obj)
            _objs.append(obj)
        return _objs

    def create_objs_observables(self, params):
        def _num_objs_observable_callable(_):
            return params['num_objs']

        def _radius_observable_callable(_):
            return params['radius']

        def _height_observable_callable(_):
            return params['height']

        return {
            'num_objs': observable.Generic(_num_objs_observable_callable),
            'radius':   observable.Generic(_radius_observable_callable),
            'height':   observable.Generic(_height_observable_callable),
        }
