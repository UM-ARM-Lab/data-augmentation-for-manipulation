import pathlib
from typing import Dict, List

import tensorflow as tf

from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from state_space_dynamics.base_dynamics_function import DynamicsEnsemble


class UDNNEnsembleWithRobot(DynamicsEnsemble):

    def __init__(self, path: pathlib.Path, elements: List, batch_size: int, scenario: ScenarioWithVisualization):
        self.constants_keys = []
        super().__init__(path=path, elements=elements, batch_size=batch_size, scenario=scenario,
                         constants_keys=self.constants_keys)
        self.state_keys += ['joint_positions']
        self.state_metadata_keys += ['joint_names']

        # NOTE: we need collision checking to be on here, because the FastRobotFeasibilityChecker relies on
        #  checking whether the tool positions match the actions.
        self.j = scenario.robot.jacobian_follower
        assert self.j.is_collision_checking()

    def propagate_from_example(self, inputs: Dict, training: bool):
        inputs_for_net = {}
        for k, v in inputs.items():
            if k in ['scene_msg', 'joint_positions', 'joint_names', 'kinect_params', 'kinect_pose']:
                continue
            inputs_for_net[k] = v

        mean, stdev = self.ensemble(self.element_class.propagate_from_example, inputs_for_net, training)

        reached, joint_positions, joint_names = self.scenario.follow_jacobian_from_example(inputs, j=self.j)

        joint_positions = tf.convert_to_tensor(joint_positions, dtype=tf.float32)

        mean['joint_positions'] = joint_positions
        mean['joint_names'] = joint_names

        zero = tf.zeros_like(joint_positions)
        stdev['joint_positions'] = zero
        stdev['joint_names'] = zero
        return mean, stdev
