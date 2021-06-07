import pathlib
from typing import Dict, List

from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from moonshine.ensemble import Ensemble2
from moonshine.filepath_tools import load_params
from moonshine.moonshine_utils import sequence_of_dicts_to_dict_of_tensors, add_batch, remove_batch, \
    dict_of_sequences_to_sequence_of_dicts_tf, numpify


class BaseDynamicsFunction:

    def __init__(self, path: pathlib.Path, batch_size: int, scenario: ScenarioWithVisualization):
        self.scenario = scenario
        self.batch_size = batch_size
        self.path = path
        self.hparams = load_params(path.parent)
        self.dynamics_dataset_hparams = self.hparams['dynamics_dataset_hparams']
        self.data_collection_params = self.dynamics_dataset_hparams['data_collection_params']
        self.max_step_size = self.data_collection_params['max_step_size']
        self.state_keys = self.hparams['state_keys']
        self.state_metadata_keys = self.hparams['state_metadata_keys']
        self.action_keys = self.hparams['action_keys']

    def propagate(self, environment: Dict, start_state: Dict, actions: List[Dict]):
        return numpify(self.propagate_tf(environment, start_state, actions))

    def propagate_tf(self, environment: Dict, start_state: Dict, actions: List[Dict]):
        predictions_dict = remove_batch(self.propagate_tf_batched(*add_batch(environment, start_state, actions)))
        predictions_list = dict_of_sequences_to_sequence_of_dicts_tf(predictions_dict)
        return predictions_list

    def propagate_tf_batched(self, environment: Dict, start_state: Dict, actions: List[Dict]):
        net_inputs = self.batched_args_to_dict(actions, environment, start_state)
        predictions = self.propagate_from_example(net_inputs, training=False)
        return predictions

    def batched_args_to_dict(self, actions, environment, start_state):
        net_inputs = {}
        start_state_with_time = add_batch(start_state, batch_axis=0)  # add time dimension of size 1
        net_inputs.update(start_state_with_time)
        net_inputs.update(environment)
        net_inputs.update(sequence_of_dicts_to_dict_of_tensors(actions, axis=1))
        # net_inputs = add_batch(net_inputs)  # having this this seems wrong...

        return net_inputs

    def propagate_from_example(self, inputs: Dict, training: bool):
        raise NotImplementedError()


class DynamicsEnsemble(BaseDynamicsFunction):
    def __init__(self, path, elements, batch_size, scenario, constants_keys: List[str]):
        self.ensemble = Ensemble2(elements, constants_keys)
        m0 = self.ensemble.elements[0]
        self.element_class = m0.__class__

        self.path = path
        self.batch_size = batch_size
        self.scenario = scenario
        self.hparams = load_params(path.parent)
        self.dynamics_dataset_hparams = m0.hparams['dynamics_dataset_hparams']
        self.data_collection_params = m0.dynamics_dataset_hparams['data_collection_params']
        self.max_step_size = m0.data_collection_params['max_step_size']
        self.state_keys = m0.state_keys
        self.state_metadata_keys = m0.state_metadata_keys
        self.action_keys = m0.action_keys

    def propagate_tf(self, *args, **kwargs):
        mean, stdev = self.ensemble(self.element_class.propagate_tf_batched, *args, **kwargs)
        return mean, stdev

    def propagate_tf_batched(self, *args, **kwargs):
        mean, stdev = self.ensemble(self.element_class.propagate_tf_batched, *args, **kwargs)
        return mean, stdev

    def propagate_from_example(self, *args, **kwargs):
        mean, stdev = self.ensemble(self.element_class.propagate_from_example, *args, **kwargs)
        return mean, stdev
