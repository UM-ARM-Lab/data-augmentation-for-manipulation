import pathlib
from typing import List, Optional

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_classifiers.feasibility_checker import FeasibilityChecker, NewFeasibilityChecker
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
    _, common_hparams = load_trial(path.parent.absolute())
    if scenario is None:
        scenario_name = common_hparams['scenario']
        scenario = get_scenario(scenario_name)
    model_type = common_hparams['model_class']
    if model_type == 'rnn':
        return NNClassifierWrapper(path, batch_size=1, scenario=scenario)
    elif model_type == 'collision':
        return PointsCollisionChecker(path, scenario=scenario)
    elif model_type == 'gripper_distance':
        return GripperDistanceChecker(path, scenario=scenario)
    elif model_type == 'feasibility':
        return FeasibilityChecker(path, scenario=scenario)
    elif model_type == 'new_feasibility':
        return NewFeasibilityChecker(path, scenario=scenario)
    else:
        raise NotImplementedError("invalid model type {}".format(model_type))
