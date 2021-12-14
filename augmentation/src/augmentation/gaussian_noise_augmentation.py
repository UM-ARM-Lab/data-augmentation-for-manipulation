from typing import Dict
import numpy as np


class GaussianNoiseAugmentation:

    def __init__(self, params):
        self.params = params
        self.rng = np.random.RandomState(0)

    def aug_opt(self, example: Dict, batch_size: int, time: int):
        example_out = {}
        for k, v in example.items():
            if k in self.params:
                std = self.params[k]
                mean = 0
                noise = self.rng.rand(*v.shape) * std + mean
                v_out = v + noise
            else:
                v_out = v
            example_out[k] = v_out
        return example_out
