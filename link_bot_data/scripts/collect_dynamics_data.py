#!/usr/bin/env python
import argparse
import pathlib

from arc_utilities import ros_init
from link_bot_data.base_collect_dynamics_data import collect_dynamics_data
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(0.1)


@ros_init.with_ros("collect_dynamics_data")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("collect_dynamics_params", type=pathlib.Path, help="json file with envrionment parameters")
    parser.add_argument("n_trajs", type=int, help='how many trajectories to collect')
    parser.add_argument("nickname")
    parser.add_argument("--seed", '-s', type=int, help='seed')
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument('--save-format', choices=['pkl', 'h5', 'tfrecord'], default='pkl')

    args = parser.parse_args()

    collect_dynamics_data(**vars(args))


if __name__ == '__main__':
    main()
