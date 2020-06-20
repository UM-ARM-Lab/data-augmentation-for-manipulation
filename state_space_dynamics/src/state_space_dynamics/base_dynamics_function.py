import json
import pathlib
from typing import Dict, List

import numpy as np
import tensorflow as tf

from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.params import CollectDynamicsParams


class BaseDynamicsFunction:

    def __init__(self, model_dir: pathlib.Path, batch_size: int, scenario: ExperimentScenario):
        model_hparams_file = model_dir / 'hparams.json'
        if not model_hparams_file.exists():
            model_hparams_file = model_dir.parent / 'params.json'
            if not model_hparams_file.exists():
                raise FileNotFoundError("no hparams file found!")
        self.scenario = scenario
        self.hparams = json.load(model_hparams_file.open('r'))
        self.batch_size = batch_size
        data_params = CollectDynamicsParams.from_json(self.hparams['dynamics_dataset_hparams']['data_collection_params'])
        self.dt = data_params.dt
        self.max_step_size = data_params.max_step_size
        self.states_description = self.hparams['dynamics_dataset_hparams']['states_description']
        self.states_keys = None
        self.action_keys = None

    def propagate_from_example(self, dataset_element):
        raise NotImplementedError()

    # TODO: make propagate use the "environment" dict abstraction
    def propagate(self,
                  environment: Dict,
                  start_states: Dict[str, np.ndarray],
                  actions: np.ndarray) -> List[Dict]:
        for k in start_states.keys():
            start_states[k] = start_states[k].astype(np.float32)
        actions = actions.astype(np.float32)

        actions = tf.Variable(actions, dtype=tf.float32, name='actions')
        predictions = self.propagate_differentiable(environment,
                                                    start_states,
                                                    actions)
        predictions_np = []
        for state_t in predictions:
            state_np = {}
            for k, v in state_t.items():
                state_np[k] = v.numpy().astype(np.float64)
            predictions_np.append(state_np)

        return predictions_np

    def propagate_differentiable(self,
                                 environment: Dict,
                                 start_states: Dict[str, np.ndarray],
                                 actions: tf.Variable) -> List[Dict]:
        raise NotImplementedError()
