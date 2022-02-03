from typing import Dict

import numpy as np

from augmentation.train_test_aug_vae import load_model_artifact, PROJECT
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from moonshine.vae import MyVAE, reparametrize


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
        x = self.scenario.example_dict_to_flat_vector(example)

        h = self.model.encoder(x)
        mu = h[..., :int(self.model.hparams.latent_dim)]
        log_var = h[..., int(self.model.hparams.latent_dim):]
        hidden = reparametrize(mu, log_var / self.temperature)
        x_reconstruction = self.model.decoder(hidden)

        example_aug = self.scenario.flat_vector_to_example_dict(example, x_reconstruction)
        return example_aug
