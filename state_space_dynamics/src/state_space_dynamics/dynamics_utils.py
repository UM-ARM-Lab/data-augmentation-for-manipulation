import pathlib
from typing import Tuple, List, Optional

from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.get_scenario import get_scenario
from moonshine.filepath_tools import load_trial
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction
from state_space_dynamics.image_cond_dyn import ImageCondDynamicsWrapper
from state_space_dynamics.unconstrained_dynamics_nn import UDNNWrapper
from state_space_dynamics.udnn_with_robot import UDNNWithRobotKinematicsWrapper


def load_generic_model(model_dirs: List[pathlib.Path],
                       scenario: Optional[ExperimentScenario] = None) -> Tuple[BaseDynamicsFunction, Tuple[str]]:
    # FIXME: remove batch_size=1 here? can I put it in base model?
    # we use the first model and assume they all have the same hparams
    representative_model_dir = model_dirs[0]
    _, common_hparams = load_trial(representative_model_dir.parent.absolute())

    if scenario is None:
        scenario_name = common_hparams['dynamics_dataset_hparams']['scenario']
        scenario = get_scenario(scenario_name)

    model_class = common_hparams['model_class']
    if model_class == 'SimpleNN':
        nn = UDNNWrapper(model_dirs, batch_size=1, scenario=scenario)
        return nn, representative_model_dir.parts[1:]
    elif model_class == 'UDNN+Robot':
        nn = UDNNWithRobotKinematicsWrapper(model_dirs, batch_size=1, scenario=scenario)
        return nn, representative_model_dir.parts[1:]
    elif model_class == 'ImageCondDyn':
        nn = ImageCondDynamicsWrapper(model_dirs, batch_size=1, scenario=scenario)
        return nn, representative_model_dir.parts[1:]
    else:
        raise NotImplementedError("invalid model type {}".format(model_class))
