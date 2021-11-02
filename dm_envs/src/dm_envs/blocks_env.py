from typing import Dict

from dm_control.composer.observation import observable
from dm_control.manipulation.props.primitive import Box
from dm_control.manipulation.shared import arenas, constants, robots
from dm_control.manipulation.shared import registry, tags
from dm_control.manipulation.shared.observations import VISION

from dm_envs.planar_pushing import PlanarPushingTask
from dm_envs.primitive_hand import PrimitiveHand


class PlanarPushingBlocksTask(PlanarPushingTask):

    def create_objs(self, params):
        _blocks = []
        for i in range(params['num_blocks']):
            block = Box(half_lengths=[params['block_size'] / 2] * 3, name=f'block{i}')
            self._arena.add_free_entity(block)
            _blocks.append(block)
        return _blocks

    def create_objs_observables(self, params):
        def _num_blocks_observable_callable(_):
            return params['num_blocks']

        def _block_size_observable_callable(_):
            return params['block_size']

        return {
            'num_blocks': observable.Generic(_num_blocks_observable_callable),
            'block_size': observable.Generic(_block_size_observable_callable),
        }
