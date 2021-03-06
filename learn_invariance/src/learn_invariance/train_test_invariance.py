import pathlib
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import transformations
from tqdm import tqdm

from learn_invariance.invariance_model import InvarianceModel, compute_transformation_invariance_error
from learn_invariance.invariance_model_wrapper import InvarianceModelWrapper
from link_bot_data.new_base_dataset import NewBaseDatasetLoader
from merrrt_visualization.rviz_animation_controller import RvizSimpleStepper
from moonshine import common_train_hparams
from moonshine.filepath_tools import load_hjson
from moonshine.model_runner import ModelRunner
from moonshine.torch_and_tf_utils import remove_batch
from state_space_dynamics.train_test_dynamics_tf import setup_training_paths


def train_main(dataset_dirs: List[pathlib.Path],
               model_hparams: pathlib.Path,
               log: str,
               batch_size: int,
               epochs: int,
               seed: int,
               checkpoint: Optional[pathlib.Path] = None,
               no_validate: bool = False,
               trials_directory: pathlib.Path = pathlib.Path('trials'),
               **kwargs):
    model_hparams = load_hjson(model_hparams)

    train_dataset_loader = NewBaseDatasetLoader(dataset_dirs=dataset_dirs)
    train_dataset = train_dataset_loader.get_datasets('train').batch(batch_size).shuffle()
    val_dataset_loader = NewBaseDatasetLoader(dataset_dirs=dataset_dirs)
    val_dataset = val_dataset_loader.get_datasets('val').batch(batch_size)

    model_hparams.update(common_train_hparams.setup_hparams(batch_size, dataset_dirs, seed, train_dataset_loader))
    model = InvarianceModel(hparams=model_hparams, batch_size=batch_size, scenario=train_dataset_loader.get_scenario())

    trial_path = setup_training_paths(checkpoint, log, model_hparams, trials_directory)

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
    dataset_loader = NewBaseDatasetLoader(dataset_dirs=dataset_dirs)
    dataset = dataset_loader.batch(batch_size=1).shuffle()

    s = dataset_loader.get_scenario()
    m = InvarianceModelWrapper(checkpoint, batch_size=1, scenario=s)

    stepper = RvizSimpleStepper()
    for i, inputs in enumerate(dataset):
        transformation = inputs['transformation']
        true_error = compute_transformation_invariance_error(inputs, s)
        predicted_error = m.evaluate(transformation)
        predicted_error = predicted_error.numpy().squeeze()
        transformation = remove_batch(transformation)

        print(transformation, true_error, predicted_error)

        transform_matrix = transformations.compose_matrix(translate=transformation[:3], angles=transformation[3:])
        s.tf.send_transform_matrix(transform_matrix, parent='world', child='viz_transform')
        s.plot_error_rviz(predicted_error)
        # stepper.step()


def viz_eval(m, transformation):
    predicted_error = m.evaluate(transformation)
    predicted_error = predicted_error.numpy().squeeze()
    return predicted_error


def dim_viz_main(checkpoint: pathlib.Path, **kwargs):
    plt.style.use("slides")

    m = InvarianceModelWrapper(checkpoint, batch_size=1)

    n = 100

    fig, axes = plt.subplots(1, 2, sharey=True)
    axes[0].set_ylim([0, 1])
    axes[1].set_ylim([0, 1])
    axes[0].axhline(0.1, color='black', linestyle='--')
    axes[1].axhline(0.1, color='black', linestyle='--')
    fig.suptitle("Predicted Error of Augmentation")

    plot_angle_invariance(axes[0], m, n)
    plot_position_invariance(axes[1], m, n)

    plt.show()


lim = np.array([0.5, 0.5, 0.5, np.pi, np.pi, np.pi])


def plot_angle_invariance(ax, m, n):
    ax.set_xlabel(f"rotation (deg)")
    ax.set_ylabel("error")

    def _plot_by_axis(axis: str):
        if axis == 'roll':
            param_idx = 3
        elif axis == 'pitch':
            param_idx = 4
        elif axis == 'yaw':
            param_idx = 5
        else:
            raise NotImplementedError(axis)

        transformation_params = np.random.uniform(-lim, lim, size=[n, 6]).astype(np.float32)
        for i in [3, 4, 5]:
            if i != param_idx:
                transformation_params[:, i] = 0
        transformation_params[:, 0] = 0
        transformation_params[:, 1] = 0
        transformation_params[:, 2] = 0
        angles = transformation_params[:, param_idx]
        errors = viz_eval(m, transformation_params)
        angles_deg = np.rad2deg(angles)

        ax.scatter(angles_deg, errors, label=axis)

    _plot_by_axis("roll")
    _plot_by_axis("pitch")
    _plot_by_axis("yaw")
    ax.legend()


def plot_position_invariance(ax, m, n):
    ax.set_xlabel(f"translation (m)")
    ax.set_ylabel("error")

    def _plot_by_axis(axis: str):
        if axis == 'x':
            param_idx = 0
        elif axis == 'y':
            param_idx = 1
        elif axis == 'z':
            param_idx = 2
        else:
            raise NotImplementedError(axis)

        transformation_params = np.random.uniform(-lim, lim, size=[n, 6]).astype(np.float32)
        for i in [0, 1, 2]:
            if i != param_idx:
                transformation_params[:, i] = 0
        transformation_params[:, 3] = 0
        transformation_params[:, 4] = 0
        transformation_params[:, 5] = 0
        positions = transformation_params[:, param_idx]
        errors = viz_eval(m, transformation_params)

        ax.scatter(positions, errors, label=axis)

    _plot_by_axis("x")
    _plot_by_axis("y")
    _plot_by_axis("z")
    ax.legend()


def sorted_by_errror():
    l = []
    for e in tqdm(train_dataset):
        for e_i, t_i in zip(e['error'], e['transform']):
            l.append(np.concatenate([[e_i.numpy()], np.abs(t_i.numpy())]))
    l_np = np.array(l)

    high_trans = l_np[np.where(np.all(np.abs(l_np[:, 0:3]) > 0.05, axis=1))]
    high_trans_and_small_roll_or_pitch = l_np[np.where(
        np.logical_and(np.all(np.abs(l_np[:, 0:3]) > 0.05, axis=1), np.all(np.abs(l_np[:, 4:6]) < 0.2, axis=1)))]

    high_roll_or_pitch = l_np[np.where(np.all(np.abs(l_np[:, 4:6]) > 0.05, axis=1))]

    small_roll_or_pitch = l_np[np.where(np.all(np.abs(l_np[:, 4:6]) < 0.05, axis=1))]

    import matplotlib.pyplot as plt
    plt.scatter()
