import pathlib
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import transformations

from learn_invariance.invariance_model import InvarianceModel
from learn_invariance.invariance_model_wrapper import InvarianceModelWrapper
from learn_invariance.new_dynamics_dataset_loader import NewDynamicsDatasetLoader
from merrrt_visualization.rviz_animation_controller import RvizSimpleStepper
from moonshine import common_train_hparams
from moonshine.filepath_tools import load_hjson
from moonshine.model_runner import ModelRunner
from moonshine.moonshine_utils import remove_batch
from state_space_dynamics.train_test import setup_training_paths


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

    train_dataset_loader = NewDynamicsDatasetLoader(dataset_dirs=dataset_dirs)
    train_dataset = train_dataset_loader.get_dataset(mode='train').batch(batch_size).shuffle()
    val_dataset_loader = NewDynamicsDatasetLoader(dataset_dirs=dataset_dirs)
    val_dataset = val_dataset_loader.get_dataset(mode='val').batch(batch_size)

    model_hparams.update(common_train_hparams.setup_hparams(batch_size, dataset_dirs, seed, train_dataset_loader))
    model = InvarianceModel(hparams=model_hparams, batch_size=batch_size, scenario=train_dataset_loader.get_scenario())

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


def viz_main(dataset_dirs: List[pathlib.Path],
             checkpoint: pathlib.Path,
             mode: str,
             **kwargs,
             ):
    dataset_loader = NewDynamicsDatasetLoader(dataset_dirs=dataset_dirs)
    dataset = dataset_loader.get_dataset(mode=mode).batch(batch_size=1).shuffle()

    s = dataset_loader.get_scenario()
    m = InvarianceModelWrapper(checkpoint, batch_size=1, scenario=s)

    stepper = RvizSimpleStepper()
    for i, inputs in enumerate(dataset):
        transformation = inputs['transformation']
        predicted_error = m.evaluate(transformation)
        predicted_error = predicted_error.numpy().squeeze()
        transformation = remove_batch(transformation)
        transform_matrix = transformations.compose_matrix(translate=transformation[:3], angles=transformation[3:])
        s.tf.send_transform_matrix(transform_matrix, parent='world', child='viz_transform')
        s.plot_error_rviz(predicted_error)
        stepper.step()


def viz_eval(m, transformation):
    predicted_error = m.evaluate(transformation)
    predicted_error = predicted_error.numpy().squeeze()
    return predicted_error


def dim_viz_main(checkpoint: pathlib.Path, **kwargs):
    plt.style.use("slides")

    m = InvarianceModelWrapper(checkpoint, batch_size=1)

    n = 100

    plot_angle_invariance(m, n)

    plot_position_invariance(m, n)

    plt.show()


def plot_angle_invariance(m, n):
    plt.figure()
    axes = plt.gca()
    plt.title("Predicted Augmentation Error")
    plt.xlabel(f"rotation (deg)")
    plt.ylabel("error")

    angles = np.linspace(-np.pi / 2, np.pi / 2, n, dtype=np.float32)

    def _plot_by_axis(axis: str):
        if axis == 'roll':
            param_idx = 3
        elif axis == 'pitch':
            param_idx = 4
        elif axis == 'yaw':
            param_idx = 5
        else:
            raise NotImplementedError(axis)

        transformation_params = np.zeros([n, 6], dtype=np.float32)
        transformation_params[:, param_idx] = angles
        errors = viz_eval(m, transformation_params)
        angles_deg = np.rad2deg(angles)

        axes.scatter(angles_deg, errors, label=axis)

    _plot_by_axis("roll")
    _plot_by_axis("pitch")
    _plot_by_axis("yaw")
    plt.legend()


def plot_position_invariance(m, n):
    plt.figure()
    axes = plt.gca()
    plt.title("Predicted Augmentation Error")
    plt.xlabel(f"translation (m)")
    plt.ylabel("error")

    positions = np.linspace(-0.5, 0.5, n, dtype=np.float32)

    def _plot_by_axis(axis: str):
        if axis == 'x':
            param_idx = 0
        elif axis == 'y':
            param_idx = 1
        elif axis == 'z':
            param_idx = 2
        else:
            raise NotImplementedError(axis)

        transformation_params = np.zeros([n, 6], dtype=np.float32)
        transformation_params[:, param_idx] = positions
        errors = viz_eval(m, transformation_params)

        axes.scatter(positions, errors, label=axis)

    _plot_by_axis("x")
    _plot_by_axis("y")
    _plot_by_axis("z")
    plt.legend()
