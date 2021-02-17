#!/usr/bin/env python
import argparse
import logging
import pathlib
from typing import Optional

import colorama
import tensorflow as tf

from arc_utilities import ros_init
from link_bot_planning.test_scenes import get_states_to_save, save_test_scene
from link_bot_pycommon.args import my_formatter


@ros_init.with_ros("save_as_test_scene")
def main():
    colorama.init(autoreset=True)
    tf.get_logger().setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("scenes_dir", type=pathlib.Path)
    parser.add_argument("trial_idx", type=int, default=0)
    parser.add_argument("--force", '-f', action='store_true')

    args = parser.parse_args()

    save_as_test_scene(trial_idx=args.trial_idx, save_test_scenes_dir=args.scenes_dir, force=args.force)


def save_as_test_scene(trial_idx: int, save_test_scenes_dir: Optional[pathlib.Path] = None, force: bool = False):
    save_test_scenes_dir.mkdir(exist_ok=True, parents=True)

    print("Make sure you hit play")
    joint_state, links_states = get_states_to_save()

    save_test_scene(joint_state, links_states, save_test_scenes_dir, trial_idx, force)


if __name__ == '__main__':
    main()
