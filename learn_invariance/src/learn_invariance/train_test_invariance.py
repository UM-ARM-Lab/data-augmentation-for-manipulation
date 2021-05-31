import pathlib
from typing import List, Optional

from learn_invariance.invariance_model import InvarianceModel
from learn_invariance.new_dynamics_dataset_loader import NewDynamicsDatasetLoader
from moonshine import common_train_hparams
from moonshine.filepath_tools import load_hjson
from moonshine.metrics import AccuracyCheckpointMetric
from moonshine.model_runner import ModelRunner
from state_space_dynamics.train_test import setup_training_paths


def setup_hparams(batch_size, dataset_dirs, seed, train_dataset, use_gt_rope):
    hparams = common_train_hparams.setup_hparams(batch_size, dataset_dirs, seed, train_dataset, use_gt_rope)
    hparams.update({
        'classifier_dataset_hparams': train_dataset.hparams,
    })
    return hparams


def train_main(dataset_dirs: List[pathlib.Path],
               model_hparams: pathlib.Path,
               log: str,
               batch_size: int,
               epochs: int,
               seed: int,
               use_gt_rope: bool = True,
               checkpoint: Optional[pathlib.Path] = None,
               no_validate: bool = False,
               trials_directory: pathlib.Path = pathlib.Path('trials'),
               **kwargs):
    model_hparams = load_hjson(model_hparams)

    train_dataset = NewDynamicsDatasetLoader(dataset_dirs=dataset_dirs, mode='train', batch_size=batch_size, shuffle=True)
    val_dataset = NewDynamicsDatasetLoader(dataset_dirs=dataset_dirs, mode='val', batch_size=batch_size)

    model_hparams.update(setup_hparams(batch_size, dataset_dirs, seed, train_dataset, use_gt_rope))
    model = InvarianceModel(hparams=model_hparams, batch_size=batch_size, scenario=train_dataset.scenario)

    checkpoint_name, trial_path = setup_training_paths(checkpoint, log, model_hparams, trials_directory)

    if no_validate:
        mid_epoch_val_batches = None
        val_every_n_batches = None
        save_every_n_minutes = None
        validate_first = False
    else:
        mid_epoch_val_batches = 20
        val_every_n_batches = 50
        save_every_n_minutes = 20
        validate_first = True

    runner = ModelRunner(model=model,
                         training=True,
                         params=model_hparams,
                         trial_path=trial_path,
                         checkpoint=checkpoint,
                         mid_epoch_val_batches=mid_epoch_val_batches,
                         val_every_n_batches=val_every_n_batches,
                         save_every_n_minutes=save_every_n_minutes,
                         validate_first=validate_first,
                         batch_metadata={})

    final_val_metrics = runner.train(train_dataset, val_dataset, num_epochs=epochs)

    return trial_path, final_val_metrics
