import argparse
import pathlib
from time import sleep

import pyautogui
from torch.utils.data import DataLoader

from arc_utilities import ros_init
from link_bot_data.load_dataset import get_dynamics_dataset_loader
from moonshine.indexing import try_index_time
from moonshine.torch_utils import my_collate
from propnet.propnet_models import PropNet
from propnet.torch_dynamics_dataset import TorchDynamicsDataset
from propnet.train_test_propnet import PROJECT
from link_bot_pycommon.load_wandb_model import load_model_artifact
from propnet.visualization import plot_cylinders_paths
from tf.transformations import quaternion_from_euler


def viz_traj(dataset_dir, checkpoint):
    dataset = TorchDynamicsDataset(dataset_dir, 'all')
    s = dataset.get_scenario()

    loader = DataLoader(dataset, collate_fn=my_collate)

    model = load_model_artifact(checkpoint, PropNet, project=PROJECT, version='best', user='armlab')
    model.training = False

    inputs = next(iter(loader))
    gt_vel, gt_pos, pred_vel, pred_pos = model(inputs)

    b = 0
    plot_cylinders_paths(b, dataset, gt_pos, inputs, pred_pos, pred_vel, s)


@ros_init.with_ros("illustrative_example_viz")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('eval_dataset_dir', type=pathlib.Path)
    parser.add_argument('aug_dataset_dir', type=pathlib.Path)
    args = parser.parse_args()

    outdir = pathlib.Path("anims") / 'toy'
    outdir.mkdir(exist_ok=True, parents=True)

    aug_loader = get_dynamics_dataset_loader(args.aug_dataset_dir)
    aug_dataset = aug_loader.get_datasets(mode='all').skip(500).take(3)
    scenario = aug_loader.get_scenario()

    def _screenshot(filename):
        full_filename = outdir / filename
        region = (520, 100, 900, 900)
        full_filename.unlink(missing_ok=True)
        sleep(0.5)
        pyautogui.screenshot(full_filename.as_posix(), region=region)
        scenario.reset_viz()

    input("change to top-down ortho")
    viz_traj(args.eval_dataset_dir, '8baon')
    _screenshot(f"toy_no_aug.png")

    viz_traj(args.eval_dataset_dir, '4jt7a')
    _screenshot(f"toy_aug.png")

    input("change to frame-aligned")
    scenario.reset_viz()
    for _ in range(20):
        scenario.tf.send_transform([-0.24, -0.14, 0.5], quaternion_from_euler(-0.9, 0, -1.), 'world', 'anim_camera')
        sleep(0.1)

    T = 50
    for i, aug_example in enumerate(aug_dataset):

        def time_color(_t):
            alpha = (_t + 2) / (T + 20)
            color = [1, 0, 0, alpha]
            return color

        def viz_cylinders_t(_t):
            s_t = try_index_time(aug_example, aug_loader.state_keys, t=_t, inclusive=False)
            scenario.plot_state_rviz(s_t, label=f'pred{_t}', color=time_color(_t))

        scenario.reset_viz()
        for _ in range(5):
            scenario.plot_environment_rviz(aug_example)
            for t in range(0, T, 5):
                viz_cylinders_t(t)

        _screenshot(f"toy_aug_example_{i}.png")


if __name__ == '__main__':
    main()
