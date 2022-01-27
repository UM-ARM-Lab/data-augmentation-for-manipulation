#!/usr/bin/env python
import argparse
import logging
import pathlib
import pickle

import colorama
import numpy as np
import tensorflow as tf

from link_bot_planning.test_scenes import load_goal, save_goal
from link_bot_pycommon.args import my_formatter, int_set_arg


def main():
    colorama.init(autoreset=True)
    tf.get_logger().setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("goal_dirname", type=pathlib.Path)
    parser.add_argument("trial_indices", type=int_set_arg)
    parser.add_argument("--dx", type=float, default=0)
    parser.add_argument("--dy", type=float, default=0)
    parser.add_argument("--dz", type=float, default=0)
    parser.add_argument("--new-key", type=str)

    args = parser.parse_args()

    for trial_idx in args.trial_indices:
        goal = load_goal(args.goal_dirname, trial_idx)

        print(f'Trial: {trial_idx} loaded goal: {goal}')

        if 'midpoint' in goal:
            key = 'midpoint'
        elif 'point' in goal:
            key = 'point'
        else:
            raise NotImplementedError()

        p = goal[key]
        p[0] += args.dx
        p[1] += args.dy
        p[2] += args.dz

        if args.new_key:
            goal.pop(key)
            out_key = args.new_key
        else:
            out_key = key

        goal[out_key] = p.astype(np.float32)

        print("saving", goal)
        save_goal(goal, args.goal_dirname, trial_idx)


if __name__ == '__main__':
    main()
