import pathlib
from copy import deepcopy
from typing import Dict, List

import tensorflow as tf

from link_bot_data.dataset_utils import deserialize_scene_msg
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from moonshine.moonshine_utils import add_batch, remove_batch, dict_of_sequences_to_sequence_of_dicts_tf, \
    numpify, sequence_of_dicts_to_dict_of_tensors
from state_space_dynamics.base_dynamics_function import DynamicsEnsemble


class UDNNEnsembleWithRobot(DynamicsEnsemble):

    def __init__(self, path: pathlib.Path, elements: List, batch_size: int, scenario: ScenarioWithVisualization):
        super().__init__(path=path, elements=elements, batch_size=batch_size, scenario=scenario,
                         constants_keys=[])
        self.state_keys = deepcopy(self.state_keys) + ['joint_positions']
        self.state_metadata_keys = deepcopy(self.state_metadata_keys) + ['joint_names']

        # NOTE: we need collision checking to be on here, because the FastRobotFeasibilityChecker relies on
        #  checking whether the tool positions match the actions.
        self.j = scenario.robot.jacobian_follower
        assert self.j.is_collision_checking()

    def propagate(self, environment: Dict, start_state: Dict, actions: List[Dict]):
        return numpify(self.propagate_tf(environment, start_state, actions))

    def propagate_tf(self, environment: Dict, start_state: Dict, actions: List[Dict]):
        actions_dict = sequence_of_dicts_to_dict_of_tensors(actions, axis=1)
        mean_dict, stdev_dict = remove_batch(
            *self.propagate_tf_batched(*add_batch(environment, start_state, actions_dict)))
        mean_list = dict_of_sequences_to_sequence_of_dicts_tf(mean_dict)
        stdev_list = dict_of_sequences_to_sequence_of_dicts_tf(stdev_dict)
        return mean_list, stdev_list

    def propagate_tf_batched(self, environment: Dict, start_state: Dict, actions: Dict):
        net_inputs = self.batched_args_to_dict(environment, start_state, actions)
        mean, stdev = self.propagate_from_example(net_inputs, training=False)
        return mean, stdev

    def propagate_from_example(self, inputs: Dict, training: bool):
        inputs['batch_size'] = inputs['env'].shape[0]  # don't love this
        inputs_for_net = self.remove_some_keys(inputs)
        mean, stdev = self.ensemble(self.element_class.propagate_from_example, inputs_for_net, training)
        deserialize_scene_msg(inputs)
        inputs = numpify(inputs)
        reached, joint_positions, joint_names = self.scenario.follow_jacobian_from_example(inputs, j=self.j)

        stdev_t = tf.reduce_sum(tf.concat(list(stdev.values()), axis=-1), keepdims=True, axis=-1)
        mean['stdev'] = stdev_t

        joint_positions = tf.convert_to_tensor(joint_positions, dtype=tf.float32)
        mean['joint_positions'] = joint_positions
        mean['joint_names'] = joint_names
        zero = tf.zeros_like(joint_positions)
        stdev['joint_positions'] = zero
        stdev['joint_names'] = zero

        return mean, stdev

    @staticmethod
    def remove_some_keys(inputs):
        inputs_for_net = {}
        for k, v in inputs.items():
            if k in ['scene_msg', 'joint_positions', 'joint_names', 'kinect_params', 'kinect_pose']:
                continue
            inputs_for_net[k] = v
        return inputs_for_net
