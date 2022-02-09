import argparse
import pathlib
from time import sleep

import pyautogui
from torch.utils.data import DataLoader

from arc_utilities import ros_init
from moonshine.torch_datasets_utils import my_collate
from propnet.propnet_models import PropNet
from propnet.torch_dynamics_dataset import TorchDynamicsDataset
from link_bot_pycommon.load_wandb_model import load_model_artifact
from propnet.visualization import plot_cylinders_paths


@ros_init.with_ros("rollout_image_viz")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)
    parser.add_argument('checkpoint')
    parser.add_argument('--take', type=int, default=10)
    parser.add_argument('--user', '-u', type=str, default='armlab')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'val', 'all'], default='val')

    args = parser.parse_args()

    dataset = TorchDynamicsDataset(args.dataset_dir, args.mode)
    s = dataset.get_scenario()

    loader = DataLoader(dataset, collate_fn=my_collate)

    model = load_model_artifact(args.checkpoint, PropNet, 'propnet', version='best', user=args.user)
    model.training = False

    outdir = pathlib.Path('results') / 'rollout_images'
    outdir.mkdir(parents=True, exist_ok=True)

    def _screenshot(filename):
        full_filename = outdir / filename
        region = (590, 190, 800, 700)
        full_filename.unlink(missing_ok=True)
        sleep(0.8)
        pyautogui.screenshot(full_filename.as_posix(), region=region)
        s.reset_viz()

    s.reset_viz()
    viz_indices = [1, 5]
    for i, inputs in enumerate(loader):
        if i > max(viz_indices):
            break
        if i in viz_indices:
            gt_vel, gt_pos, pred_vel, pred_pos = model(inputs)
            b = 0
            for _ in range(4):
                plot_cylinders_paths(b, dataset, gt_pos, inputs, pred_pos, pred_vel, s)
            _screenshot(f"{args.checkpoint}_rollout{i}.png")


if __name__ == '__main__':
    main()
