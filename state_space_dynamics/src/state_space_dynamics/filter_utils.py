import pathlib
from typing import List

from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.get_scenario import get_scenario
from moonshine.filepath_tools import load_trial
from state_space_dynamics.base_filter_function import BaseFilterFunction, PassThroughFilter


def load_filter(model_dirs: List[pathlib.Path], scenario: ExperimentScenario = None) -> BaseFilterFunction:
    representative_model_dir = model_dirs[0]
    _, common_hparams = load_trial(representative_model_dir.parent.absolute())
    if scenario is None:
        scenario_name = common_hparams['dynamics_dataset_hparams']['scenario']
        scenario = get_scenario(scenario_name)
    model_type = common_hparams['model_class']
    if model_type in ['none', 'pass-through']:
        return PassThroughFilter()
    else:
        raise NotImplementedError("invalid model type {}".format(model_type))
