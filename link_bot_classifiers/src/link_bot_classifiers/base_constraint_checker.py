import pathlib
from typing import Dict, List, Optional

from link_bot_pycommon.experiment_scenario import ExperimentScenario


class BaseConstraintChecker:

    def __init__(self, paths: List[pathlib.Path], scenario: ExperimentScenario):
        self.paths = paths
        self.scenario = scenario
        self.horizon = 1
        self.hparams = {}

    def check_constraint_from_example(self, example: Dict, training: Optional[bool] = False):
        raise NotImplementedError()

    def check_constraint(self,
                         environment: Dict,
                         states_sequence: List[Dict],
                         actions: List[Dict]):
        raise NotImplementedError()

    def check_constraint_tf(self,
                            environment: Dict,
                            states_sequence: List[Dict],
                            actions: List[Dict]):
        raise NotImplementedError()
