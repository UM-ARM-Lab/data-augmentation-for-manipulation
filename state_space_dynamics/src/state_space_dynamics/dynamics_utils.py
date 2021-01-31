import pathlib
from typing import Tuple, List

from link_bot_pycommon.get_scenario import get_scenario
from my_cfm.cfm import CFMLatentDynamics
from moonshine.filepath_tools import load_trial
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction
from state_space_dynamics.image_cond_dyn import ImageCondDynamicsWrapper
from state_space_dynamics.unconstrained_dynamics_nn import UDNNWrapper


def load_generic_model(model_dirs: List[pathlib.Path]) -> Tuple[BaseDynamicsFunction, Tuple[str]]:
    # FIXME: remove batch_size=1 here? can I put it in base model?
    # we use the first model and assume they all have the same hparams
    representative_model_dir = model_dirs[0]
    _, common_hparams = load_trial(representative_model_dir.parent.absolute())
    scenario_name = common_hparams['dynamics_dataset_hparams']['scenario']
    scenario = get_scenario(scenario_name)
    model_type = common_hparams['model_class']
    if model_type == 'SimpleNN':
        nn = UDNNWrapper(model_dirs, batch_size=1, scenario=scenario)
        return nn, representative_model_dir.parts[1:]
    elif model_type == 'ImageCondDyn':
        nn = ImageCondDynamicsWrapper(model_dirs, batch_size=1, scenario=scenario)
        return nn, representative_model_dir.parts[1:]
    elif model_type == 'CFM':
        nn = CFMLatentDynamics(model_dirs, batch_size=1, scenario=scenario)
        return nn, representative_model_dir.parts[1:]
    else:
        raise NotImplementedError("invalid model type {}".format(model_type))
