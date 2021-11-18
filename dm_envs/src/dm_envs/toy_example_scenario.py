from time import sleep
from typing import Dict, Optional

import numpy as np

from dm_envs.cylinders_scenario import CylindersScenario
from dm_envs.planar_pushing_scenario import get_tcp_pos, ACTION_Z
from dm_envs.toy_example_task import ToyExampleTask
from link_bot_pycommon.bbox_visualization import viz_action_sample_bbox
from link_bot_pycommon.experiment_scenario import get_action_sample_extent, is_out_of_bounds


class ToyExampleScenario(CylindersScenario):

    def sample_action(self,
                      action_rng: np.random.RandomState,
                      environment: Dict,
                      state: Dict,
                      action_params: Dict,
                      validate, stateless: Optional[bool] = False):
        viz_action_sample_bbox(self.gripper_bbox_pub, get_action_sample_extent(action_params))

        start_gripper_position = get_tcp_pos(state)

        action_dict = {
            'gripper_position': start_gripper_position,
        }

        # first check if any objects are wayyy to far
        num_objs = state['num_objs'][0]
        for i in range(num_objs):
            obj_position = state[f'obj{i}/position'][0]
            out_of_bounds = is_out_of_bounds(obj_position, action_params['extent'])
            if out_of_bounds:
                return action_dict, (invalid := True)  # this will cause the current trajectory to be thrown out

        dx = action_rng.uniform(0, action_params['max_distance_gripper_can_move'])

        gripper_position = np.array([start_gripper_position[0] + dx, start_gripper_position[1], ACTION_Z])

        self.tf.send_transform(gripper_position, [0, 0, 0, 1], 'world', 'sample_action_gripper_position')

        action_dict['gripper_position'] = gripper_position

        return action_dict, (invalid := False)

    def make_dm_task(self, params):
        return ToyExampleTask(params)
