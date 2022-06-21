#!/usr/bin/env python
import argparse
import pathlib

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation

from learn_invariance.new_dynamics_dataset import NewDynamicsDatasetLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)
    parser.add_argument('mode', type=str, choices=['train', 'val', 'test'])

    args = parser.parse_args()

    # load the dataset
    loader = NewDynamicsDatasetLoader([args.dataset_dir])
    dataset = loader.get_datasets(mode=args.mode)

    for example in dataset:
        # matplotlib animation showing the cylinders moving
        fig = plt.figure()
        plt.axis("equal")
        plt.title(f"Trajectory #{example['traj_idx']}")
        ax = plt.gca()
        ax.set_xlim([-.2, .2])

        def viz_t(t):
            while len(ax.patches) > 0:
                ax.patches.pop()
            radius = example['radius'][t, 0]
            x = example['jaco_arm/primitive_hand/tcp_pos'][t, 0, 0]
            y = example['jaco_arm/primitive_hand/tcp_pos'][t, 0, 1]
            dx = example['jaco_arm/primitive_hand/tcp_vel'][t, 0, 0]
            dy = example['jaco_arm/primitive_hand/tcp_vel'][t, 0, 1]
            robot = patches.Circle((x, y), radius, color='pink')
            robot_vel = patches.Arrow(x, y, dx, dy, width=0.01, color='red')
            ax.add_patch(robot)
            ax.add_patch(robot_vel)

            for object_idx in range(9):
                x = example[f'obj{object_idx}/position'][t, 0, 0]
                y = example[f'obj{object_idx}/position'][t, 0, 1]
                dx = example[f'obj{object_idx}/linear_velocity'][t, 0, 0]
                dy = example[f'obj{object_idx}/linear_velocity'][t, 0, 1]

                obj = plt.Circle((x, y), radius, color='blue')
                obj_vel = plt.Arrow(x, y, dx, dy, width=0.01, color='black')
                ax.add_patch(obj)
                ax.add_patch(obj_vel)

            ax.set_ylim([-.2, .2])

        anim = FuncAnimation(fig, viz_t, frames=50, interval=2)
        # anim.save()
        plt.show()


if __name__ == '__main__':
    main()
