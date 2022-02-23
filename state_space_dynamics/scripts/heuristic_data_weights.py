#!/usr/bin/env python
import argparse
import pathlib
from typing import Dict

import numpy as np
import tensorflow as tf

from arc_utilities import ros_init
from link_bot_data.modify_dataset import modify_dataset2
from link_bot_data.new_base_dataset import NewBaseDatasetLoader
from link_bot_data.split_dataset import split_dataset_via_files
from link_bot_pycommon.collision_checking import batch_in_collision_tf_3d
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(None)


@ros_init.with_ros("heuristic_data_weights")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')

    args = parser.parse_args()

    outdir = args.dataset_dir.parent / f"{args.dataset_dir.name}+heuristic-weights"
    dataset = NewBaseDatasetLoader([args.dataset_dir])
    scenario = dataset.get_scenario()

    def _process_example(dataset, example: Dict):
        points = scenario.state_to_points_for_cc(example)
        inflation = float(tf.squeeze(example['res']))
        in_collision, inflated_env = batch_in_collision_tf_3d(environment=example, points=points,
                                                              inflate_radius_m=inflation)
        weight = 1 - in_collision.numpy().astype(np.float32)
        # scenario.plot_environment_rviz({'env': inflated_env, 'res': example['res'], 'origin_point': example['origin_point']})
        # scenario.plot_environment_rviz(example)
        # scenario.plot_points_rviz(tf.reshape(points, [-1, 3]).numpy(), label='cc', scale=0.005)
        weight_padded = np.concatenate((weight, [1]))
        weight = np.logical_and(weight_padded[:-1], weight_padded[1:]).astype(np.float32)
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
