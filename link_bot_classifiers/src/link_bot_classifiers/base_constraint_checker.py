import pathlib
from typing import Dict, List, Optional

from link_bot_pycommon.experiment_scenario import ExperimentScenario
from moonshine.filepath_tools import load_params


class BaseConstraintChecker:

    def __init__(self, path: pathlib.Path, scenario: ExperimentScenario):
        self.path = path
        self.scenario = scenario
        self.horizon = 2
        self.hparams = load_params(self.path.parent)
        self.name = self.__class__.__name__

    def check_constraint_from_example(self, example: Dict, training: Optional[bool] = False):
        raise NotImplementedError()

    def check_constraint_tf_batched(self,
                                    environment: Dict,
                                    states: Dict,
                                    actions: Dict,
                                    batch_size: int,
                                    state_sequence_length: int):
        raise NotImplementedError()

    def check_constraint_tf(self,
                            environment: Dict,
                            states_sequence: List[Dict],
                            actions: List[Dict]):
        raise NotImplementedError()

    def check_constraint(self,
                         environment: Dict,
                         states_sequence: List[Dict],
                         actions: List[Dict]):
        c = self.check_constraint_tf(environment, states_sequence, actions)
        return c.numpy()
