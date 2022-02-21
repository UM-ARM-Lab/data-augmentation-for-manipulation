import pathlib
from typing import Dict, List

import tensorflow as tf
import tensorflow.keras.layers as layers

from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from moonshine.tensorflow_utils import vector_to_dict, sequence_of_dicts_to_dict_of_tensors
from moonshine.my_keras_model import MyKerasModel
from moonshine.restore_model import restore_model
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


class UnconstrainedDynamicsNN(MyKerasModel):

    def __init__(self, hparams: Dict, batch_size: int, scenario: ExperimentScenario):
        super().__init__(hparams=hparams, batch_size=batch_size)
        self.scenario = scenario
        self.initial_epoch = 0

        self.concat = layers.Concatenate()
        self.dense_layers = []
        for fc_layer_size in self.hparams['fc_layer_sizes']:
            self.dense_layers.append(layers.Dense(fc_layer_size, activation='relu', use_bias=True))

        self.state_keys: List = self.hparams['state_keys']
        self.action_keys: List = self.hparams['action_keys']
        dataset_state_description: Dict = self.hparams['dynamics_dataset_hparams']['state_description']
        self.dataset_action_description: Dict = self.hparams['dynamics_dataset_hparams']['action_description']
        self.state_description = {k: dataset_state_description[k] for k in self.state_keys}
        self.total_state_dimensions = sum([dataset_state_description[k] for k in self.state_keys])

        self.dense_layers.append(layers.Dense(self.total_state_dimensions, activation=None))

    @tf.function
    def call(self, example, training, mask=None):
        actions = {k: example[k] for k in self.action_keys}
        input_sequence_length = actions[self.action_keys[0]].shape[1]
        s_0 = {k: example[k][:, 0] for k in self.state_keys}

        pred_states = [s_0]
        for t in range(input_sequence_length):
            s_t = pred_states[-1]
            action_t = {k: example[k][:, t] for k in self.action_keys}
            local_action_t = self.scenario.put_action_local_frame(s_t, action_t)

            s_t_local = self.scenario.put_state_local_frame(s_t)
            states_and_actions = list(s_t_local.values()) + list(local_action_t.values())

            # concat into one big state-action vector
            z_t = self.concat(states_and_actions)
            for dense_layer in self.dense_layers:
                z_t = dense_layer(z_t)

            delta_s_t = vector_to_dict(self.state_description, z_t)
            s_t_plus_1 = self.scenario.integrate_dynamics(s_t, delta_s_t)

            pred_states.append(s_t_plus_1)

        pred_states_dict = sequence_of_dicts_to_dict_of_tensors(pred_states, axis=1)
        return pred_states_dict

    def compute_loss(self, example, outputs):
        return {
            'loss': self.scenario.dynamics_loss_function(example, outputs)
        }

    def compute_metrics(self, metrics, dataset_element, outputs):
        metrics = self.scenario.dynamics_metrics_function(dataset_element, outputs)
        metrics['loss'] = self.scenario.dynamics_loss_function(dataset_element, outputs)
        return metrics


class UDNNWrapper(BaseDynamicsFunction):

    def __init__(self, path: pathlib.Path, batch_size: int, scenario: ScenarioWithVisualization):
        super().__init__(path, batch_size, scenario)
        self.net = UnconstrainedDynamicsNN(hparams=self.hparams, batch_size=batch_size, scenario=scenario)
        restore_model(self.net, path)

    def propagate_from_example(self, inputs: Dict, training: bool):
        return self.net(self.net.preprocess_no_gradient(inputs, training), training)
