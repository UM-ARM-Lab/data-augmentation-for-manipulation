import pathlib

import numpy as np

from link_bot_classifiers.knn_recovery_policy import KNNRecoveryPolicy
from link_bot_classifiers.nn_recovery_policy import NNRecoveryPolicy
from link_bot_classifiers.random_recovery_policy import RandomRecoveryPolicy
from link_bot_classifiers.simple_recovery_policy import SimpleRecoveryPolicy
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from shape_completion_training.model.filepath_tools import load_trial


def load_generic_model(model_dir: pathlib.Path, scenario: ExperimentScenario, rng: np.random.RandomState):
    _, hparams = load_trial(model_dir.parent.absolute())

    model_class = hparams['model_class']
    if model_class == 'simple':
        return SimpleRecoveryPolicy(hparams, model_dir, scenario, rng)
    elif model_class == 'random':
        return RandomRecoveryPolicy(hparams, model_dir, scenario, rng)
    elif model_class == 'nn':
        return NNRecoveryPolicy(hparams, model_dir, scenario, rng)
    elif model_class == 'knn':
        return KNNRecoveryPolicy(hparams, model_dir, scenario, rng)
    else:
        raise NotImplementedError(f"model type {model_class} not implemented")
