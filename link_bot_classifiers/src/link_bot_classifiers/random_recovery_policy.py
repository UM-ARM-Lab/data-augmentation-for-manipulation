import pathlib
import numpy as np
from typing import Dict
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_classifiers.base_recovery_policy import BaseRecoveryPolicy


class RandomRecoveryPolicy(BaseRecoveryPolicy):

    def __init__(self, hparams: Dict, model_dir: pathlib.Path, scenario: ExperimentScenario,
                 rng: np.random.RandomState):
        super().__init__(hparams, model_dir, scenario, rng)

    def __call__(self, environment: Dict, state: Dict):
        action, _ = self.scenario.sample_action(action_rng=self.rng, environment=environment, state=state,
                                           action_params=self.hparams, validate=True)
        return action
