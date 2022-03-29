#!/usr/bin/env python
import argparse
import pathlib
from typing import Dict

from arc_utilities import ros_init
from arc_utilities.filesystem_utils import no_overwrite_path
from link_bot_data.modify_dataset import modify_dataset2
from link_bot_data.new_base_dataset import NewBaseDatasetLoader
from moonshine.robot_points_tf import RobotVoxelgridInfo
from link_bot_data.split_dataset import split_dataset_via_files
from moonshine.gpu_config import limit_gpu_mem
from state_space_dynamics.heuristic_data_weights import heuristic_weight_func

limit_gpu_mem(None)


@ros_init.with_ros("heuristic_data_weights")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('-n', '--nickname', help='nickname', default='')

    args = parser.parse_args()

    hparams = {
        'heuristic_weighting': True,  # don't change this, it's just metadata
        'env_inflation':       0.9,
        'check_robot':         True,
        'robot_inflation':     0.6,
        'max_rope_length':     0.774,
        'check_length':        False,
    }

    outdir = args.dataset_dir.parent / f"{args.dataset_dir.name}+{args.nickname}heuristic-weights"
    outdir = no_overwrite_path(outdir)
    print(outdir)
    dataset = NewBaseDatasetLoader([args.dataset_dir])
    scenario = dataset.get_scenario()

    robot_points_path = pathlib.Path("robot_points_data/val_high_res/robot_points.pkl")
    robot_info = RobotVoxelgridInfo('joint_positions', robot_points_path)

    def _process_example(dataset, example: Dict):
        yield from heuristic_weight_func(scenario, example, hparams, robot_info)

    modify_dataset2(dataset_dir=args.dataset_dir,
                    dataset=dataset,
                    outdir=outdir,
                    process_example=_process_example,
                    hparams_update=hparams,
                    save_format='pkl')
    split_dataset_via_files(outdir, 'pkl')
    print(outdir)


if __name__ == '__main__':
    main()
