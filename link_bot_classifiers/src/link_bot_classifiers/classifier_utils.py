import pathlib
from typing import Optional

import numpy as np

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker, ConstraintCheckerEnsemble
from link_bot_classifiers.feasibility_checker import RobotFeasibilityChecker, FastRobotFeasibilityChecker
from link_bot_classifiers.gripper_distance_checker import GripperDistanceChecker
from link_bot_classifiers.nn_classifier import NNClassifierWrapper
from link_bot_classifiers.points_collision_checker import PointsCollisionChecker
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.get_scenario import get_scenario
from moonshine.filepath_tools import load_trial


def load_generic_model(path: pathlib.Path,
                       scenario: Optional[ExperimentScenario] = None) -> BaseConstraintChecker:
    # FIXME: remove batch_size=1 here? can I put it in base model?
    # we use the first model and assume they all have the same hparams
    _, params = load_trial(path.parent.absolute())
    if scenario is None:
        scenario_name = params['scenario']
        scenario = get_scenario(scenario_name)
    model_type = params['model_class']
    if model_type == 'rnn':
        return NNClassifierWrapper(path, batch_size=1, scenario=scenario)
    elif model_type == 'ensemble':
        const_keys_for_classifier = []
        models = [load_generic_model(pathlib.Path(checkpoint)) for checkpoint in params['checkpoints']]
        ensemble = ConstraintCheckerEnsemble(path, models, const_keys_for_classifier)
        return ensemble
    elif model_type == 'collision':
        return PointsCollisionChecker(path, scenario=scenario)
    elif model_type == 'gripper_distance':
        return GripperDistanceChecker(path, scenario=scenario)
    elif model_type == 'feasibility':
        return RobotFeasibilityChecker(path, scenario=scenario)
    elif model_type == 'new_feasibility':
        # return FastRobotFeasibilityChecker(path, scenario=scenario)
        return FastRobotFeasibilityChecker(path, scenario=scenario)
    else:
        raise NotImplementedError("invalid model type {}".format(model_type))


def make_max_class_prob(probabilities):
    other_class_probabilities = 1 - probabilities
    return np.maximum(probabilities, other_class_probabilities)
