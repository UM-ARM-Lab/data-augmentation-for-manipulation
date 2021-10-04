#!/usr/bin/env python
import argparse
import pathlib

from colorama import Fore

from arc_utilities import ros_init
from link_bot_classifiers.eval_proxy_datasets import eval_proxy_datasets
from moonshine.filepath_tools import load_hjson


@ros_init.with_ros("eval_ift_classifier_metrics")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ift_dirs', type=pathlib.Path, nargs='+')

    args = parser.parse_args()

    checkpoints = []
    for ift_dir in args.ift_dirs:
        for planning_results_dir in (ift_dir / 'planning_results').iterdir():
            metadata = load_hjson(planning_results_dir / 'metadata.hjson')
            checkpoint = pathlib.Path(metadata['planner_params']['classifier_model_dir'][0])
            if not checkpoint.exists():
                print(Fore.RED + f"checkpoint {checkpoint.as_posix()} does not exist!" + Fore.RESET)
            elif 'untrained-1' in checkpoint.as_posix():
                print(f"Skipping {checkpoint.as_posix()}")
                pass
            else:
                checkpoints.append(checkpoint)

    checkpoints = list(set(checkpoints))

    print(Fore.GREEN + "Evaluating the following checkpoints:" + Fore.RESET)
    for c in checkpoints:
        print(c.as_posix())

    eval_proxy_datasets(checkpoints=checkpoints)


if __name__ == '__main__':
    main()
