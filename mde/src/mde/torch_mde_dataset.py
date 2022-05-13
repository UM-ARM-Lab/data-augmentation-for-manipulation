import pathlib

from link_bot_data.dataset_utils import add_predicted
from link_bot_data.visualization import classifier_transition_viz_t, init_viz_env, plot_state_t
from moonshine.indexing import index_time_batched, index_time
from moonshine.my_torch_dataset import MyTorchDataset
from moonshine.numpify import numpify
from moonshine.torch_and_tf_utils import remove_batch


class TorchMDEDataset(MyTorchDataset):

    def __init__(self, dataset_dir: pathlib.Path, mode: str, transform=None, only_metadata=False, is_empty=False):
        super().__init__(dataset_dir, mode, transform, only_metadata, is_empty)
        self.model_hparams = self.params['fwd_model_hparams']
        self.data_collection_params = self.params['data_collection_params']
        self.scenario_params = self.data_collection_params['scenario_params']
        self.state_description = self.data_collection_params['state_description']
        self.predicted_state_keys = [add_predicted(k) for k in self.model_hparams['state_keys']]
        self.state_metadata_description = self.data_collection_params['state_metadata_description']
        self.action_description = self.data_collection_params['action_description']
        self.env_description = self.data_collection_params['env_description']
        self.state_keys = list(self.state_description.keys())
        self.state_metadata_keys = list(self.state_metadata_description.keys())
        self.env_keys = list(self.env_description.keys())
        self.action_keys = list(self.action_description.keys())
        self.time_indexed_keys = self.state_keys + self.state_metadata_keys + self.action_keys
        self.time_indexed_keys_predicted = self.predicted_state_keys + self.state_metadata_keys + self.action_keys

    def index_time_batched(self, example_batched, t: int):
        e_t = numpify(remove_batch(index_time_batched(example_batched, self.time_indexed_keys, t, False)))
        return e_t

    def index_time(self, example, t: int):
        e_t = numpify(index_time(example, self.time_indexed_keys, t, False))
        return e_t

    def transition_viz_t(self):
        return classifier_transition_viz_t(metadata={},
                                           state_metadata_keys=self.state_metadata_keys,
                                           predicted_state_keys=self.predicted_state_keys,
                                           true_state_keys=self.state_keys + ['error'])

    def viz_pred_actual(self, actual, pred):
        init_viz_env(self.get_scenario(), actual)
        plot_state_t(self.get_scenario(), self.state_keys, actual, t=0, label='actual_0', color='r')
        plot_state_t(self.get_scenario(), self.state_keys, actual, t=1, label='actual_1', color='r')
        plot_state_t(self.get_scenario(), self.state_keys, pred, t=0, label='pred_0', color='b')
        plot_state_t(self.get_scenario(), self.state_keys, pred, t=1, label='pred_1', color='b')
