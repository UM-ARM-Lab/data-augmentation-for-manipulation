#!/usr/bin/env python
import argparse
import pathlib

from arc_utilities import ros_init
from link_bot_data.new_dataset_utils import fetch_mde_dataset
from link_bot_data.visualization import init_viz_env
from link_bot_pycommon.pycommon import print_dict
from mde.torch_mde_dataset import TorchMDEDataset
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.gpu_config import limit_gpu_mem
from moonshine.torch_datasets_utils import dataset_take, dataset_shard, dataset_skip

limit_gpu_mem(None)


def visualize_dataset(dataset, take, skip, shard, threshold=None):
    print_dict(dataset[0])

    s = dataset.get_scenario()

    dataset_ = dataset_take(dataset, take)
    dataset_ = dataset_skip(dataset_, skip)
    dataset_ = dataset_shard(dataset_, shard)

    dataset_anim = RvizAnimationController(n_time_steps=len(dataset_), ns='trajs')
    time_anim = RvizAnimationController(n_time_steps=2)

    n_examples_visualized = 0
    while not dataset_anim.done:
        inputs = dataset_[dataset_anim.t()]
        print(inputs['example_idx'])

        time_anim.reset()
        while not time_anim.done:
            t = time_anim.t()
            init_viz_env(s, inputs, t)
            dataset.transition_viz_t()(s, inputs, t)
            error_t = inputs['error'][t]

            if threshold is not None:
                is_close = error_t < threshold
                s.plot_is_close(is_close)
            time_anim.step()

            n_examples_visualized += 1

        dataset_anim.step()

    print(f"{n_examples_visualized:=}")


@ros_init.with_ros("visualize_mde_dataset")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)
    parser.add_argument('--mode', choices=['train', 'val', 'test', 'all'], default='all')
    parser.add_argument('--shard', type=int)
    parser.add_argument('--threshold', type=float, default=0.06)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--take', type=int)

    args = parser.parse_args()

    dataset = TorchMDEDataset(fetch_mde_dataset(args.dataset_dir), mode=args.mode)
    visualize_dataset(dataset, args.take, args.skip, args.shard, threshold=args.threshold)


if __name__ == '__main__':
    main()
