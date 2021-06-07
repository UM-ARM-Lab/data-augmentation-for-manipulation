import pathlib
from typing import Dict, List

import numpy as np
import tensorflow as tf

from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.pycommon import n_state_to_n_points
from moonshine.moonshine_utils import dict_of_sequences_to_sequence_of_dicts, sequence_of_dicts_to_dict_of_sequences
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


class RigidTranslationModel(BaseDynamicsFunction):

    def __init__(self, path: List[pathlib.Path], batch_size: int, scenario: ExperimentScenario):
        super().__init__(path, batch_size, scenario)
        assert len(path) == 1
        self.batch_size = batch_size
        b = self.hparams['B']
        self.B = tf.constant(np.array(b), dtype=tf.float32)
        self.states_keys = self.hparams['states_keys']

    def propagate_from_example(self, dataset_element, training: bool = False):
        inputs, _ = dataset_element
        batch_states = {key: inputs[key] for key in self.states_keys}
        batch_states = dict_of_sequences_to_sequence_of_dicts(batch_states)
        predictions = {k: [] for k in self.states_keys}
        for full_env, full_env_origin, res, states, actions in zip(inputs['full_env/env'],
                                                                   inputs['full_env/origin'],
                                                                   inputs['full_env/res'],
                                                                   batch_states,
                                                                   inputs['action']):
            start_states = {k: states[k][0] for k in states.keys()}
            out_states = self.propagate_tf(full_env=full_env,
                                           full_env_origin=full_env_origin,
                                           res=res,
                                           start_state=start_states,
                                           actions=actions)
            out_states = sequence_of_dicts_to_dict_of_sequences(out_states)
            for k in self.states_keys:
                predictions[k].append(out_states[k])
        predictions = {k: tf.stack(predictions[k], axis=0) for k in predictions.keys()}
        return predictions

    def propagate_tf(self, environment: Dict, start_state: Dict, actions: List[Dict]) -> List[Dict]:
        del environment  # unused
        net_inputs = {k: tf.expand_dims(start_state[k], axis=0) for k in self.state_keys}
        net_inputs.update(sequence_of_dicts_to_dict_of_tensors(actions))
        net_inputs = add_batch(net_inputs)
        net_inputs = make_dict_tf_float32(net_inputs)
        # the network returns a dictionary where each value is [T, n_state]
        # which is what you'd want for training, but for planning and execution and everything else
        # it is easier to deal with a list of states where each state is a dictionary
        predictions = self.net(net_inputs, training=False)
        predictions = remove_batch(predictions)
        predictions = dict_of_sequences_to_sequence_of_dicts_tf(predictions)
        return predictions
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        s_t = {}
        for k, s_0_k in start_state.items():
            s_t[k] = tf.convert_to_tensor(s_0_k, dtype=tf.float32)
        predictions = [s_t]
        for t in range(actions.shape[0]):
            action_t = actions[t]

            s_t_plus_1 = {}
            for k, s_t_k in s_t.items():
                n_points = n_state_to_n_points(s_t_k.shape[0])
                delta_s_t = tf.tensordot(action_t, tf.transpose(self.B), axes=1)
                delta_s_t_flat = tf.tile(delta_s_t, [n_points])
                s_t_k = s_t_k + delta_s_t_flat * self.dt
                s_t_plus_1[k] = s_t_k

            predictions.append(s_t_plus_1)
            s_t = s_t_plus_1
        return predictions
