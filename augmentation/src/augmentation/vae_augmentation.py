from typing import Dict

import numpy as np
import torch

from augmentation.train_test_aug_vae import load_model_artifact, PROJECT
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from moonshine.vae import MyVAE


class VAEAugmentation:

    def __init__(self, scenario: ScenarioWithVisualization, model_path):
        self.model = load_model_artifact(model_path, model_class=MyVAE, project=PROJECT, version='best', user='armlab')
        self.scenario = scenario
        self.rng = np.random.RandomState(0)

    def aug_opt(self, example: Dict, batch_size: int, time: int):
        x = torch.tensor(self.scenario.example_dict_to_flat_vector(example))
        x_aug = self.model(x)
        example_aug = self.scenario.flat_vector_to_aug_example_dict(example, x_aug)
        return example_aug