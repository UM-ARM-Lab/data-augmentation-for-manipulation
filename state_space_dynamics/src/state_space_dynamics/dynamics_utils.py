import pathlib
from functools import lru_cache
from typing import Optional

from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.get_scenario import get_scenario
from moonshine.filepath_tools import load_trial
from state_space_dynamics.udnn_with_robot import UDNNEnsembleWithRobot
from state_space_dynamics.unconstrained_dynamics_nn import UDNNWrapper


@lru_cache
def load_generic_model(model_dir: pathlib.Path, scenario: Optional[ExperimentScenario] = None):
    _, params = load_trial(model_dir.parent.absolute())

    if scenario is None:
        scenario_name = params['dynamics_dataset_hparams']['scenario']
        scenario = get_scenario(scenario_name)

    model_class = params['model_class']
    if model_class == 'SimpleNN':
        nn = UDNNWrapper(model_dir, batch_size=1, scenario=scenario)
        return nn
    elif model_class == 'udnn_ensemble_with_robot':
        elements = [load_generic_model(pathlib.Path(checkpoint), scenario) for checkpoint in params['rope_dynamics_checkpoints']]
        nn = UDNNEnsembleWithRobot(model_dir, elements, batch_size=1, scenario=scenario)
        return nn
    else:
        raise NotImplementedError("invalid model type {}".format(model_class))
