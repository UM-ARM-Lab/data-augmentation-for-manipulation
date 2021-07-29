#!/usr/bin/env python
import tensorflow as tf
import argparse
import numpy as np
import pathlib
from typing import Dict

from tqdm import tqdm

from link_bot_pycommon.grid_utils import occupied_voxels_to_points
from link_bot_pycommon.job_chunking import JobChunker
from link_bot_pycommon.serialization import load_gzipped_pickle
from tensorflow_graphics.nn.loss.chamfer_distance import evaluate

from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(None)


def cd_env_dist(aug_env, data_env):
    # chamfer distances
    subsample = 10
    aug_points = occupied_voxels_to_points(aug_env, 0.02, [0, 0, 0])[::subsample]
    data_points = occupied_voxels_to_points(data_env, 0.02, [0, 0, 0])[::subsample]
    chamfer_distance = evaluate(aug_points, data_points)
    return chamfer_distance


def compute_distance(aug_example: Dict, data_example: Dict):
    aug_rope = aug_example['rope']
    aug_rope_before = aug_rope[0]
    aug_rope_after = aug_rope[1]
    aug_rope_before_points = tf.reshape(aug_rope_before, [-1, 3])
    aug_rope_after_points = tf.reshape(aug_rope_after, [-1, 3])
    aug_joint_positions = aug_example['joint_positions']
    aug_joint_positions_before = aug_joint_positions[0]
    aug_joint_positions_after = aug_joint_positions[1]
    aug_env = aug_example['env']
    data_rope = data_example['rope']
    data_rope_before = data_rope[0]
    data_rope_after = data_rope[1]
    data_rope_before_points = tf.reshape(data_rope_before, [-1, 3])
    data_rope_after_points = tf.reshape(data_rope_after, [-1, 3])
    data_joint_positions = data_example['joint_positions']
    data_joint_positions_before = data_joint_positions[0]
    data_joint_positions_after = data_joint_positions[1]
    data_env = data_example['env']

    rope_before_dist = tf.linalg.norm(aug_rope_before_points - data_rope_before_points)
    rope_after_dist = tf.linalg.norm(aug_rope_after_points - data_rope_after_points)

    joint_positions_before_dist = tf.linalg.norm(aug_joint_positions_before - data_joint_positions_before)
    joint_positions_after_dist = tf.linalg.norm(aug_joint_positions_after - data_joint_positions_after)

    env_dist = cd_env_dist(aug_env, data_env)

    weights = tf.constant([
        1.0,
        1.0,
        0.1,
        0.1,
        10.0,
    ])
    distances = tf.stack([
        rope_before_dist,
        rope_after_dist,
        joint_positions_before_dist,
        joint_positions_after_dist,
        env_dist,
    ], axis=0)
    combined_distance = tf.tensordot(weights, distances, axes=1)
    return combined_distance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('augdir', type=pathlib.Path)
    parser.add_argument('datadir', type=pathlib.Path)
    parser.add_argument('--regenerate', action='store_true')

    args = parser.parse_args()

    augfiles = list(args.augdir.glob("*.pkl.gz"))
    datafiles = list(args.datadir.glob("*.pkl.gz"))

    # TODO: batch on GPU
    logfilename = pathlib.Path('results') / f"{args.augdir.parent.name}-{args.datadir.parent.name}.hjson"
    jc = JobChunker(logfilename)

    for i, augfile in enumerate(tqdm(augfiles)):
        for j, datafile in enumerate(tqdm(datafiles, leave=False, position=1)):

            key = f"{i}-{j}"
            d = jc.get_result(key)
            if d is None or args.regenerate:
                try:
                    aug_example = load_gzipped_pickle(augfile)
                    data_example = load_gzipped_pickle(datafile)
                    d = compute_distance(aug_example, data_example)
                    jc.store_result(key, d)
                except Exception:
                    print("Error on ", augfile.as_posix(), datafile.as_posix())


if __name__ == '__main__':
    main()
