import pathlib
import tensorflow as tf
from colorama import Fore

from learn_invariance.invariance_model import InvarianceModel
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from moonshine.filepath_tools import load_params


def restore_model(model, path):
    ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=1)
    status = ckpt.restore(manager.latest_checkpoint).expect_partial()
    if manager.latest_checkpoint:
        print(Fore.CYAN + "Restored from {}".format(manager.latest_checkpoint) + Fore.RESET)
        if manager.latest_checkpoint:
            status.assert_nontrivial_match()
    else:
        raise RuntimeError(f"Failed to restore {manager.latest_checkpoint}!!!")


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
