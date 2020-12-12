#!/usr/bin/env python
import argparse
import logging
import pathlib
import time

import colorama
import hjson
import rospkg
import tensorflow as tf

from arc_utilities import ros_init
from link_bot_planning.full_stack_runner import FullStackRunner, run_steps
from link_bot_pycommon.args import my_formatter

r = rospkg.RosPack()


def main():
    colorama.init(autoreset=True)
    tf.get_logger().setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("full_stack_param", type=pathlib.Path)
    parser.add_argument("--gui", action='store_true')
    parser.add_argument("--launch", action='store_true')
    parser.add_argument("--steps", help="a comma separated list of steps to explicitly include, regardless of logfile")
    parser.add_argument('--verbose', '-v', action='count', default=0)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--nickname', type=str)
    group.add_argument('--logfile', type=pathlib.Path)

    args = parser.parse_args()

    ros_init.rospy_and_cpp_init("run_full_stack")

    if args.nickname is None:
        logfile_name = args.logfile
        with logfile_name.open("r") as logfile:
            log = hjson.loads(logfile.read())
        nickname = log['nickname']
        unique_nickname = f"{nickname}_{int(time.time())}"
    else:
        nickname = args.nickname
        unique_nickname = f"{nickname}_{int(time.time())}"
        # create a logfile
        logfile_dir = pathlib.Path("log") / unique_nickname
        logfile_dir.mkdir(parents=True)
        logfile_name = logfile_dir / "logfile.hjson"
        log = {'nickname': nickname}

    log['filename'] = logfile_name

    with args.full_stack_param.open('r') as f:
        full_stack_params = hjson.load(f)

    fsr = FullStackRunner(nickname=nickname,
                          unique_nickname=unique_nickname,
                          full_stack_params=full_stack_params,
                          launch=args.launch,
                          verbose=args.verbose)
    fsr.gui = args.gui
    if args.steps is not None:
        included_steps = args.steps.split(",")
    else:
        included_steps = None

    run_steps(fsr, full_stack_params, included_steps, log)

    ros_init.shutdown()


if __name__ == '__main__':
    main()
