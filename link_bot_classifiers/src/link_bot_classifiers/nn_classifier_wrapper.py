import pathlib
from typing import Dict, Optional, List

import tensorflow as tf
from colorama import Fore

from link_bot_classifiers import nn_classifier, nn_classifier2
from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_data.dataset_utils import add_predicted
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from moonshine.moonshine_utils import add_batch, sequence_of_dicts_to_dict_of_tensors, remove_batch


class NNClassifierWrapper(BaseConstraintChecker):
    def __init__(self, path: pathlib.Path, batch_size: int, scenario: ScenarioWithVisualization):
        super().__init__(path, scenario)
        self.name = self.__class__.__name__

        self.dataset_labeling_params = self.hparams['classifier_dataset_hparams']['labeling_params']
        self.data_collection_params = self.hparams['classifier_dataset_hparams']['data_collection_params']
        self.horizon = self.dataset_labeling_params['classifier_horizon']

        net_class_name = self.get_net_class()

        self.net = net_class_name(hparams=self.hparams, batch_size=batch_size, scenario=scenario)

        ckpt = tf.train.Checkpoint(model=self.net)
        manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=1)

        status = ckpt.restore(manager.latest_checkpoint).expect_partial()
        if manager.latest_checkpoint:
            print(Fore.CYAN + "Restored from {}".format(manager.latest_checkpoint) + Fore.RESET)
            if manager.latest_checkpoint:
                status.assert_nontrivial_match()
        else:
            raise RuntimeError(f"Failed to restore {manager.latest_checkpoint}!!!")

        self.state_keys = self.net.state_keys
        self.state_metadata_keys = self.net.state_metadata_keys
        self.action_keys = self.net.action_keys
        self.true_state_keys = self.net.true_state_keys
        self.pred_state_keys = self.net.pred_state_keys

    # @tf.function
    def check_constraint_from_example(self, example: Dict, training: Optional[bool] = False):
        example_preprocessed = self.net.preprocess_no_gradient(example, training)
        predictions = self.net(example_preprocessed, training=training)
        return predictions

    def check_constraint_tf_batched(self,
                                    environment: Dict,
                                    states: Dict,
                                    actions: Dict,
                                    batch_size: int,
                                    state_sequence_length: int):
        # construct network inputs
        net_inputs = {}
        if 'scene_msg' in environment:
            environment.pop('scene_msg')
        net_inputs.update(environment)

        for action_key in self.action_keys:
            net_inputs[action_key] = actions[action_key]

        for state_metadata_key in self.state_metadata_keys:
            net_inputs[state_metadata_key] = states[state_metadata_key]

        for state_key in self.state_keys:
            planned_state_key = add_predicted(state_key)
            net_inputs[planned_state_key] = states[state_key]

        if self.hparams['stdev']:
            net_inputs[add_predicted('stdev')] = states['stdev']

        net_inputs['batch_size'] = batch_size
        net_inputs['time'] = state_sequence_length
        predictions = self.check_constraint_from_example(net_inputs, training=False)
        probability = predictions['probabilities']
        probability = tf.squeeze(probability, axis=2)
        return probability

    def check_constraint_tf(self,
                            environment: Dict,
                            states_sequence: List[Dict],
                            actions: List[Dict]):
        environment = add_batch(environment)
        states_sequence_dict = sequence_of_dicts_to_dict_of_tensors(states_sequence)
        states_sequence_dict = add_batch(states_sequence_dict)
        state_sequence_length = len(states_sequence)
        actions_dict = sequence_of_dicts_to_dict_of_tensors(actions)
        actions_dict = add_batch(actions_dict)
        probabilities = self.check_constraint_tf_batched(environment=environment,
                                                         states=states_sequence_dict,
                                                         actions=actions_dict,
                                                         batch_size=1,
                                                         state_sequence_length=state_sequence_length)
        probabilities = remove_batch(probabilities)
        return probabilities

    def check_constraint(self,
                         environment: Dict,
                         states_sequence: List[Dict],
                         actions: List[Dict]):
        probabilities = self.check_constraint_tf(environment=environment,
                                                 states_sequence=states_sequence,
                                                 actions=actions)
        probabilities = probabilities.numpy()
        return probabilities

    @staticmethod
    def get_net_class():
        return nn_classifier.NNClassifier


class NNClassifier2Wrapper(NNClassifierWrapper):

    @staticmethod
    def get_net_class():
        return nn_classifier2.NNClassifier
