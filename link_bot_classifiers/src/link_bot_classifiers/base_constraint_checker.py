import pathlib
from typing import Dict, List, Optional

from link_bot_pycommon.experiment_scenario import ExperimentScenario
from moonshine.ensemble import Ensemble2
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


class ConstraintCheckerEnsemble(BaseConstraintChecker):
    def __init__(self, path, elements, constants_keys: List[str]):
        self.ensemble = Ensemble2(elements, constants_keys)
        m0 = self.ensemble.elements[0]
        self.element_class = m0.__class__
        self.threshold = 0.15
        BaseConstraintChecker.__init__(self, path, m0.scenario)

    def check_constraint_tf(self, *args, **kwargs):
        mean, stdev = self.ensemble(self.element_class.check_constraint_tf, *args, **kwargs)
        return stdev > self.threshold

    def check_constraint_tf_batched(self, *args, **kwargs):
        mean, stdev = self.ensemble(self.element_class.check_constraint_tf_batched, *args, **kwargs)
        return stdev > self.threshold

    def check_constraint_from_example(self, *args, **kwargs):
        mean, stdev = self.ensemble(self.element_class.check_constraint_from_example, *args, **kwargs)
        return stdev > self.threshold
