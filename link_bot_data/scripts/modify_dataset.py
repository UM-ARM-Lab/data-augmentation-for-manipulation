#!/usr/bin/env python
import argparse
import pathlib
import shutil

import tensorflow as tf
from tqdm import tqdm

from arc_utilities import ros_init
from link_bot_data.tf_dataset_utils import pkl_write_example
from link_bot_pycommon.collision_checking import batch_in_collision_tf_3d
from moonshine.geometry_tf import densify_points
from moonshine.my_torch_dataset import MyTorchDataset


def process_example(args):
    i, dataset, scenario, outdir = args
    example = dataset[i]

    scenario = dataset.get_scenario({'rope_name': 'rope_3d_alt'})

    points = example['rope'].reshape(10, 25, 3)
    points_dense = densify_points(10, points, 5)
    in_collision, _ = batch_in_collision_tf_3d(environment=example,
                                               points=points_dense,
                                               inflate_radius_m=0.019)
    any_in_collision = bool(tf.reduce_any(in_collision).numpy())

    scenario.plot_points_rviz(tf.reshape(points_dense, [-1, 3]), label='cc_points')

    if not any_in_collision:
        pkl_write_example(outdir, example, example['metadata']['example_idx'])
    else:
        print(f"Filtered out example {example['metadata']['example_idx']}")


@ros_init.with_ros("modify_dataset")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('suffix', type=str, help='string added to the new dataset name')

    args = parser.parse_args()

    outdir = args.dataset_dir.parent / f"{args.dataset_dir.name}+{args.suffix}"
    outdir.mkdir(exist_ok=True)

    shutil.copy(args.dataset_dir / 'hparams.hjson', outdir)
    shutil.copy(args.dataset_dir / 'train.txt', outdir)
    shutil.copy(args.dataset_dir / 'val.txt', outdir)
    shutil.copy(args.dataset_dir / 'test.txt', outdir)

    dataset = MyTorchDataset(args.dataset_dir, mode='all', no_update_with_metadata=True)
    scenario = dataset.get_scenario({'rope_name': 'rope_3d_alt'})

    for i in tqdm(range(len(dataset))):
        process_example((i, dataset, scenario, outdir))

    # with Pool() as pool:
    #     tasks = [(i, dataset, scenario, outdir) for i in range(len(dataset))]
    #     for _ in tqdm(pool.imap_unordered(process_example, tasks), total=len(tasks)):
    #         pass


if __name__ == '__main__':
    main()
