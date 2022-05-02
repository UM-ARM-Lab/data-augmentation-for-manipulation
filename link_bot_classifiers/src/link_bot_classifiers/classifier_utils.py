import pathlib
from typing import Optional

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker, ConstraintCheckerEnsemble
from link_bot_classifiers.feasibility_checker import RobotFeasibilityChecker, FastRobotFeasibilityChecker
from link_bot_classifiers.gripper_distance_checker import GripperDistanceChecker
from link_bot_classifiers.nn_classifier_wrapper import NNClassifierWrapper
from link_bot_classifiers.points_collision_checker import PointsCollisionChecker, PointsSDFCollisionChecker
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.get_scenario import get_scenario
from mde.mde_torch import MDEConstraintChecker
from moonshine.filepath_tools import load_trial


def is_torch_model(path):
    return path.as_posix().startswith('p:')


def strip_torch_model_prefix(path):
    return path.as_posix()[2:]


def load_generic_model(path: pathlib.Path,
                       scenario: Optional[ExperimentScenario] = None) -> BaseConstraintChecker:
    if is_torch_model(path):  # this is a pytorch model, not a old TF model
        return MDEConstraintChecker(strip_torch_model_prefix(path))

    _, params = load_trial(path.parent.absolute())
    if scenario is None:
        scenario_name = params['dataset_hparams']['scenario']
        scenario = get_scenario(scenario_name)
    model_type = params['model_class']
    if model_type in ['rnn', 'nn_classifier', 'nn_classifier2']:
        return NNClassifierWrapper(path, batch_size=1, scenario=scenario)
    elif model_type == 'ensemble':
        const_keys_for_classifier = []
        models = [load_generic_model(pathlib.Path(checkpoint)) for checkpoint in params['checkpoints']]
        ensemble = ConstraintCheckerEnsemble(path, models, const_keys_for_classifier)
        return ensemble
    elif model_type == 'collision':
        return PointsCollisionChecker(path, scenario=scenario)
    elif model_type == 'sdf_collision':
        return PointsSDFCollisionChecker(path, scenario=scenario)
    elif model_type == 'gripper_distance':
        return GripperDistanceChecker(path, scenario=scenario)
    elif model_type == 'feasibility':
        return RobotFeasibilityChecker(path, scenario=scenario)
    elif model_type == 'new_feasibility':
        return FastRobotFeasibilityChecker(path, scenario=scenario)
    else:
        raise NotImplementedError("invalid model type {}".format(model_type))
