import pathlib
from functools import lru_cache
from typing import Optional, Dict

import numpy as np

from arc_utilities.algorithms import nested_dict_update
from link_bot_classifiers.nn_recovery_policy import NNRecoveryPolicy, NNRecoveryEnsemble, NNRecoveryPolicy2
from link_bot_classifiers.random_recovery_policy import RandomRecoveryPolicy
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from moonshine.filepath_tools import load_trial


@lru_cache
def load_generic_model(path: pathlib.Path,
                       scenario: Optional[ScenarioWithVisualization] = None,
                       rng: np.random.RandomState = None,
                       update_hparams: Optional[Dict] = None):
    _, params = load_trial(path.parent.absolute())

    if scenario is None:
        scenario_name = params['scenario']
        scenario = get_scenario(scenario_name)

    if rng is None:
        rng = np.random.RandomState(0)

    if update_hparams is not None:
        params = nested_dict_update(params, update_hparams)

    model_class = params['model_class']
    if model_class == 'random':
        return RandomRecoveryPolicy(path, scenario, rng, update_hparams)
    elif model_class in ['nn', 'recovery']:
        return NNRecoveryPolicy(path, scenario, rng, update_hparams)
    elif model_class in ['recovery2']:
        return NNRecoveryPolicy2(path, scenario, rng, update_hparams)
    elif model_class == 'ensemble':
        const_keys_for_classifier = []
        models = [load_generic_model(pathlib.Path(checkpoint), scenario, rng, update_hparams) for checkpoint in
                  params['checkpoints']]
        ensemble = NNRecoveryEnsemble(path, models, const_keys_for_classifier, rng, update_hparams)
        return ensemble
    else:
        raise NotImplementedError(f"model type {model_class} not implemented")
