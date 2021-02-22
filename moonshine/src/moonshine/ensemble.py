import pathlib
from typing import Dict, List

import tensorflow as tf
from colorama import Fore

from link_bot_pycommon.experiment_scenario import ExperimentScenario
from moonshine.filepath_tools import load_trial
from moonshine.moonshine_utils import sequence_of_dicts_to_dict_of_tensors, flatten_after
from moonshine.my_keras_model import MyKerasModel


class Ensemble:

    def __init__(self, model_dirs: List[pathlib.Path], batch_size: int, scenario: ExperimentScenario):
        representative_model_dir = model_dirs[0]
        _, self.hparams = load_trial(representative_model_dir.parent.absolute())

        self.scenario = scenario
        self.batch_size = batch_size
        self.data_collection_params = self.hparams['dynamics_dataset_hparams']['data_collection_params']
        self.state_description = self.hparams['dynamics_dataset_hparams']['state_description']
        self.action_description = self.hparams['dynamics_dataset_hparams']['action_description']
        self.state_metadata_keys = self.hparams.get('state_metadata_keys', [])

        self.nets: List[MyKerasModel] = []
        # NOTE: this abstraction assumes everything is a NN, specifically a MyKerasModel which is not great
        for model_dir in model_dirs:
            net, ckpt = self.make_net_and_checkpoint(batch_size, scenario)
            manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=1)

            status = ckpt.restore(manager.latest_checkpoint).expect_partial()
            if manager.latest_checkpoint:
                print(Fore.CYAN + "Restored from {}".format(manager.latest_checkpoint) + Fore.RESET)
                status.assert_existing_objects_matched()
            else:
                raise RuntimeError("Failed to restore!!!")

            self.nets.append(net)

    def make_net_and_checkpoint(self, batch_size, scenario):
        """

        Args:
            batch_size:
            scenario:

        Returns:
            A callable that takes in a dictionary of batched tensors and returns an output dictionary of batched tensors
        """
        raise NotImplementedError()

    def from_example(self, example, training: bool = False):
        """ This is the function that all other functions eventually call """
        if 'batch_size' not in example:
            example['batch_size'] = self.get_batch_size(example)

        outputs = [net(net.preprocess_no_gradient(example, training), training=training) for net in self.nets]
        outputs_dict = sequence_of_dicts_to_dict_of_tensors(outputs)
        outputs_state_metadata = {k: outputs_dict[k][0] for k in self.state_metadata_keys}

        outputs_dict = {k: flatten_after(outputs_dict[k], axis=self.get_num_batch_axes()) for k in
                        self.get_output_keys()}

        # axis 0 is the different networks
        mean = {k: tf.math.reduce_mean(outputs_dict[k], axis=0) for k in self.get_output_keys()}
        mean.update(outputs_state_metadata)
        stdev = {k: tf.math.reduce_std(outputs_dict[k], axis=0) for k in self.get_output_keys()}

        # each output variable has its own vector of variances,
        # and here we sum all the elements of all the vectors to get a single scalar
        all_stdevs = tf.concat(list(stdev.values()), axis=-1)
        mean['stdev'] = tf.reduce_sum(all_stdevs, axis=-1, keepdims=True)
        return mean, stdev

    def get_batch_size(self, example: Dict):
        raise NotImplementedError()

    def get_output_keys(self):
        raise NotImplementedError()

    @staticmethod
    def get_num_batch_axes():
        raise NotImplementedError()
