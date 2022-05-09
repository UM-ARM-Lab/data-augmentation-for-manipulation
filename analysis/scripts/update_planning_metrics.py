#!/usr/bin/env python
import argparse
import pathlib

from analysis.analyze_results import load_planning_results
from analysis.results_utils import get_all_results_subdirs
from arc_utilities import ros_init
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(0.1)


@ros_init.with_ros("analyse_planning_results")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help='results directory', type=pathlib.Path,
                        default=pathlib.Path("/media/shared/planning_results"))
    parser.add_argument('--regenerate', action='store_true')

    args = parser.parse_args()

    results_dirs = get_all_results_subdirs(args.root, regenerate=args.regenerate)
    load_planning_results(results_dirs, regenerate=args.regenerate)


if __name__ == '__main__':
    main()
