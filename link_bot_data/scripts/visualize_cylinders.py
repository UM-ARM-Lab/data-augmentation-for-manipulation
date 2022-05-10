#!/usr/bin/env python
import pathlib

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation

from link_bot_data.cylinders_dataset import MyTorchDataset


def main():
    dataset_dir = pathlib.Path("/media/shared/fwd_model_data/h50+vel/")

    # load the dataset
    dataset = MyTorchDataset(dataset_dir, mode='train')

    for example in dataset:
        # matplotlib animation showing the cylinders moving
        fig = plt.figure()
        plt.axis("equal")
        plt.title(f"Trajectory #{example['traj_idx']}")
        ax = plt.gca()

        def viz_t(t):
            while len(ax.patches) > 0:
                ax.patches.pop()
            radius = example['radius'][t, 0]
            x = example['jaco_arm/primitive_hand/tcp_pos'][t, 0, 0]
            y = example['jaco_arm/primitive_hand/tcp_pos'][t, 0, 1]
            robot = patches.Circle((x, y), radius)
            ax.add_patch(robot)

            for object_idx in range(9):
                x = example[f'obj{object_idx}/position'][t, 0, 0]
                y = example[f'obj{object_idx}/position'][t, 0, 1]

                obj = plt.Circle((x, y), radius)
                ax.add_patch(obj)

            ax.set_xlim([-.2, 0.2])

        anim = FuncAnimation(fig, viz_t, frames=50, interval=2)
        # anim.save()
        plt.show()


if __name__ == '__main__':
    main()
