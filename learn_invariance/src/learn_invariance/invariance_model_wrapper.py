import pathlib
from typing import Optional

import tensorflow as tf

from learn_invariance.invariance_model import InvarianceModel
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from moonshine.filepath_tools import load_params
from moonshine.restore_model import restore_model


class InvarianceModelWrapper:

    def __init__(self, path: pathlib.Path, batch_size: int, scenario: Optional[ScenarioWithVisualization] = None):
        self.hparams = load_params(path.parent)
        if scenario is None:
            scenario = get_scenario(self.hparams['dataset_hparams']['scenario'])
        self.model = InvarianceModel(self.hparams, batch_size, scenario)
        restore_model(self.model, path)

    def evaluate(self, transformations):
        """

        Args:
            transformations: [b, 6]

        Returns: [b]

        """
        inputs = {
            'transform': transformations
        }
        outputs = self.model(inputs)
        return tf.squeeze(outputs['predicted_error'], axis=-1)
