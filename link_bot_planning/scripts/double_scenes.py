#!/usr/bin/env python
import argparse
import logging
import pathlib
import shutil

import colorama


def main():
    colorama.init(autoreset=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("dirname", type=pathlib.Path)

    args = parser.parse_args()

    n_existing_scenes = len(list(args.dirname.glob("*.bag")))

    for i in range(n_existing_scenes):
        j = i + n_existing_scenes
        print(f'{i}->{j}')

        in_scene_file = args.dirname / f'scene_{i:04d}.bag'
        out_scene_file = args.dirname / f'scene_{j:04d}.bag'

        in_goal_file = args.dirname / f'goal_{i:04d}.pkl'
        out_goal_file = args.dirname / f'goal_{j:04d}.pkl'

        if out_scene_file.exists():
            raise RuntimeError(f"file {out_scene_file.as_posix()} exists!")
        if out_goal_file.exists():
            raise RuntimeError(f"file {out_goal_file.as_posix()} exists!")

        shutil.copy(in_scene_file, out_scene_file)
        shutil.copy(in_goal_file, out_goal_file)


if __name__ == '__main__':
    main()
