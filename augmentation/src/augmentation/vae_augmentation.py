from typing import Dict

import numpy as np

from augmentation.train_test_aug_vae import load_model_artifact, PROJECT
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from moonshine.vae import MyVAE


class VAEAugmentation:

    def __init__(self, scenario: ScenarioWithVisualization, model_path):
        self.model = load_model_artifact(model_path,
                                         model_class=MyVAE,
                                         project=PROJECT,
                                         version='latest',
                                         user='armlab')
        self.model.scenario = scenario
        self.scenario = scenario
        self.rng = np.random.RandomState(0)
        self.temperature = 10

    def aug_opt(self, example: Dict, *args, **kwargs):
        example_aug = self.model(example, temperature=self.temperature)
        return example_aug
