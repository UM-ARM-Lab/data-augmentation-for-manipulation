import pathlib
from typing import Dict

import numpy as np

from arc_utilities.algorithms import nested_dict_update
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from moonshine.filepath_tools import load_params


class BaseRecoveryPolicy:

    def __init__(self,
                 path: pathlib.Path,
                 scenario: ExperimentScenario,
                 rng: np.random.RandomState,
                 update_hparams: Dict):
        self.path = path
        self.params = load_params(self.path.parent)
        self.params = nested_dict_update(self.params, update_hparams)
        self.scenario = scenario
        self.rng = rng

    def __call__(self, environment: Dict, state: Dict):
        raise NotImplementedError()
