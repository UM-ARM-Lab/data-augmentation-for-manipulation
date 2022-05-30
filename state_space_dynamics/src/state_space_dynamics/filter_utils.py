import pathlib
from typing import List

from moonshine.filepath_tools import load_trial
from state_space_dynamics.base_filter_function import BaseFilterFunction, PassThroughFilter


def load_filter(model_dirs: List[pathlib.Path], _) -> BaseFilterFunction:
    representative_model_dir = model_dirs[0]
    _, common_hparams = load_trial(representative_model_dir.parent.absolute())
    model_type = common_hparams['model_class']
    if model_type in ['none', 'pass-through']:
        return PassThroughFilter()
    else:
        raise NotImplementedError("invalid model type {}".format(model_type))
