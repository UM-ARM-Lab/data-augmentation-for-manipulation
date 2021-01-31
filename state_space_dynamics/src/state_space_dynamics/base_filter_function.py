import pathlib
from typing import Dict, List, Tuple, Optional

import numpy as np
import tensorflow as tf

from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.pycommon import make_dict_tf_float32
from moonshine.ensemble import Ensemble
from moonshine.moonshine_utils import add_batch, remove_batch, numpify
from moonshine.my_keras_model import MyKerasModel


# TODO: this is a misleading interface, a filter function doesn't need to be an ensemble of neural networks
class BaseFilterFunction(Ensemble):

    def __init__(self, model_dirs: List[pathlib.Path], batch_size: int, scenario: ExperimentScenario):
        super().__init__(model_dirs, batch_size, scenario)
        self.state_keys = self.nets[0].state_keys
        self.obs_keys = self.nets[0].obs_keys

    def filter(self, environment: Dict, state: Optional[Dict], observation: Dict) -> Tuple[Dict, Dict]:
        mean_state, stdev_state = self.filter_differentiable(environment=environment, state=state, observation=observation)
        return numpify(mean_state), numpify(stdev_state)

    def filter_differentiable(self, environment: Dict, state: Optional[Dict], observation: Dict) -> Tuple[Dict, Dict]:
        net_inputs = {}
        net_inputs.update(environment)
        net_inputs.update(observation)
        if state is not None:
            net_inputs.update(state)
        net_inputs = add_batch(net_inputs)
        net_inputs = make_dict_tf_float32(net_inputs)

        mean_state, stdev_state = self.from_example(net_inputs, training=False)
        mean_state = remove_batch(mean_state)
        stdev_state = remove_batch(stdev_state)
        return mean_state, stdev_state

    def filter_differentiable_batched(self, environment: Dict, state: Optional[Dict], observation: Dict) -> Tuple[Dict, Dict]:
        net_inputs = {}
        net_inputs.update(environment)
        net_inputs.update(observation)
        if state is not None:
            net_inputs.update(state)
        net_inputs = make_dict_tf_float32(net_inputs)
        mean_predictions, stdev_predictions = self.from_example(net_inputs, training=False)
        return mean_predictions, stdev_predictions

    def get_batch_size(self, example: Dict):
        return example[self.obs_keys[0]].shape[0]

    @staticmethod
    def get_num_batch_axes():
        return 1

    def get_output_keys(self):
        return self.state_keys + self.obs_keys

    def make_net_and_checkpoint(self, batch_size, scenario) -> Tuple[MyKerasModel, tf.train.Checkpoint]:
        raise NotImplementedError()


# TODO: fix interface definitions here
class PassThroughFilter(BaseFilterFunction):

    def __init__(self):
        pass

    def filter(self, environment: Dict, state: Optional[Dict], observation: Dict) -> Tuple[Dict, Dict]:
        mean_state = observation
        stdev_state = {k: np.zeros_like(v) for k, v in observation.items()}
        return mean_state, stdev_state
