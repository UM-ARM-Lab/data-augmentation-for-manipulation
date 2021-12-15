from typing import Dict

import numpy as np

from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization


class SimpleNoiseAugmentation:

    def __init__(self, scenario: ScenarioWithVisualization, params):
        self.params = params
        self.scenario = scenario
        self.rng = np.random.RandomState(0)

    def aug_opt(self, example: Dict, batch_size: int, time: int):
        example_out = {}
        for k, v in example.items():
            v_out = self.scenario.simple_noise(self.rng, example, k, v, self.params)
            example_out[k] = v_out
        return example_out
