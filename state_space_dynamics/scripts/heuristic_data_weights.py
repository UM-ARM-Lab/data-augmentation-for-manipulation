#!/usr/bin/env python
import tensorflow as tf
import argparse
import pathlib
from typing import Dict

from arc_utilities import ros_init
from link_bot_classifiers.points_collision_checker import get_points_for_cc
from link_bot_data.modify_dataset import modify_dataset2
from link_bot_data.new_base_dataset import NewBaseDatasetLoader
from link_bot_data.split_dataset import split_dataset_via_files
from link_bot_pycommon.collision_checking import batch_in_collision_tf_3d


@ros_init.with_ros("heuristic_data_weights")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')

    args = parser.parse_args()

    outdir = args.dataset_dir.parent / f"{args.dataset_dir.name}+heuristic-weights"
    dataset = NewBaseDatasetLoader([args.dataset_dir])
    scenario = dataset.get_scenario()

    def _process_example(dataset, example: Dict):
        points = get_points_for_cc(collision_check_object=True, scenario=scenario, state=example)
        inflation = float(tf.squeeze(example['res']) * 2)
        in_collision, _ = batch_in_collision_tf_3d(environment=example, points=points, inflate_radius_m=inflation)
        in_collision = bool(in_collision)
        # scenario.reset_viz()
        # scenario.plot_environment_rviz(example)
        # scenario.plot_points_rviz(points, label='cc')
        weight = 0 if in_collision else 1
        example['metadata']['weight'] = weight
        yield example

    hparams_update = {}

    modify_dataset2(dataset_dir=args.dataset_dir,
                    dataset=dataset,
                    outdir=outdir,
                    process_example=_process_example,
                    hparams_update=hparams_update,
                    save_format='pkl')
    split_dataset_via_files(outdir, 'pkl')
    print(outdir)


if __name__ == '__main__':
    main()
