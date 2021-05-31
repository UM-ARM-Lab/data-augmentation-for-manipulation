import pathlib

import tensorflow as tf

from learn_invariance.invariance_model import InvarianceModel
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from moonshine.filepath_tools import load_params
from moonshine.restore_model import restore_model


class InvarianceModelWrapper:

    def __init__(self, path: pathlib.Path, batch_size: int, scenario: ScenarioWithVisualization):
        self.hparams = load_params(path.parent)
        self.model = InvarianceModel(self.hparams, batch_size, scenario)
        restore_model(self.model, path)

    def evaluate(self, transformations):
        """

        Args:
            transformations: [b, 6]

        Returns: [b]

        """
        inputs = {
            'transformation': transformations
        }
        outputs = self.model(inputs)
        return tf.squeeze(outputs['predicted_error'], axis=-1)
