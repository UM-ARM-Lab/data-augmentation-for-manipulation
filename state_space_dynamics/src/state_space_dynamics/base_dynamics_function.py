import pathlib
from typing import Dict, List

from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.pycommon import make_dict_tf_float32
from moonshine.ensemble import Ensemble
from moonshine.moonshine_utils import sequence_of_dicts_to_dict_of_tensors, add_batch, remove_batch, \
    dict_of_sequences_to_sequence_of_dicts_tf, numpify


# FIXME: bad inheritance relationship
class BaseDynamicsFunction(Ensemble):

    def __init__(self, model_dirs: List[pathlib.Path], batch_size: int, scenario: ExperimentScenario):
        super().__init__(model_dirs, batch_size, scenario)
        self.max_step_size = self.data_collection_params['max_step_size']
        self.state_keys = self.nets[0].state_keys
        self.action_keys = self.nets[0].action_keys

    def propagate(self, environment: Dict, start_state: Dict, action_sequence: List[Dict]):
        mean_predictions, stdev_predictions = self.propagate_tf(environment, start_state, action_sequence)
        return numpify(mean_predictions), numpify(stdev_predictions)

    def propagate_tf(self, environment: Dict, start_state: Dict, actions: List[Dict]):
        net_inputs = {}
        # add time dimension of size 1
        start_state_with_time = add_batch(start_state, batch_axis=0)
        net_inputs.update(start_state_with_time)
        net_inputs.update(environment)
        net_inputs.update(sequence_of_dicts_to_dict_of_tensors(actions))
        net_inputs = add_batch(net_inputs)
        if 'scene_msg' in environment:
            net_inputs.pop('scene_msg')
        net_inputs = make_dict_tf_float32(net_inputs)
        # the network returns a dictionary where each value is [T, n_state]
        # which is what you'd want for training, but for planning and execution and everything else
        # it is easier to deal with a list of states where each state is a dictionary
        mean_predictions, stdev_predictions = self.from_example(net_inputs, training=False)
        mean_predictions = remove_batch(mean_predictions)
        stdev_predictions = remove_batch(stdev_predictions)
        mean_predictions = dict_of_sequences_to_sequence_of_dicts_tf(mean_predictions)
        stdev_predictions = dict_of_sequences_to_sequence_of_dicts_tf(stdev_predictions)
        return mean_predictions, stdev_predictions

    def propagate_tf_batched(self, environment: Dict, state: Dict, actions: Dict):
        net_inputs = {}
        net_inputs.update(state)
        net_inputs.update(actions)
        net_inputs.update(environment)
        mean_predictions, stdev_predictions = self.from_example(net_inputs, training=False)
        return mean_predictions, stdev_predictions

    def get_batch_size(self, example: Dict):
        return example[self.state_keys[0]].shape[0]

    @staticmethod
    def get_num_batch_axes():
        return 2

    def get_output_keys(self):
        return self.state_keys

    def make_net_and_checkpoint(self, batch_size, scenario):
        raise NotImplementedError()


class EnsembleOfDynamicsFunctions(BaseDynamicsFunction, Ensemble):
    pass
    # for model in self.dynamics_models:
    #     output_i = model(environment, state, action)
    #     {output_i[k] for k in state_keys}
    #     {output_i[k] for k in action}
    #     output_i['reached']
