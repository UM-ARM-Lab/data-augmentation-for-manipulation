from typing import Dict

import numpy as np

from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.sample_object_positions import sample_object_positions


class EmbodiedScenario(ExperimentScenario):

    def __init__(self, scenario: ExperimentScenario):
        super().__init__()
        self.scenario = scenario

    def execute_action(self, environment, state, action: Dict):
        raise NotImplementedError()

    def move_objects_randomly(self, env_rng, movable_objects_services, movable_objects, kinematic: bool,
                              timeout: float = 0.5):
        random_object_positions = sample_object_positions(env_rng, movable_objects)
        if kinematic:
            raise NotImplementedError()
        else:
            ExperimentScenario.move_objects(movable_objects_services, random_object_positions, timeout)

    def on_after_data_collection(self, params):
        pass

    def on_before_data_collection(self, params: Dict):
        pass

    def randomize_environment(self, env_rng: np.random.RandomState, params: Dict):
        raise NotImplementedError()

    def get_environment(self, params: Dict, **kwargs):
        raise NotImplementedError()

    def get_state(self):
        raise NotImplementedError()
