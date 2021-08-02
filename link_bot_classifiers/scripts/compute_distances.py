#!/usr/bin/env python
import argparse
import pathlib
from multiprocessing import Pool
from typing import Dict

import tensorflow as tf
from tensorflow_graphics.nn.loss.chamfer_distance import evaluate
from tqdm import tqdm

from link_bot_classifiers.pd_distances_utils import weights, too_far, joints_weights
from link_bot_pycommon.grid_utils import occupied_voxels_to_points
from link_bot_pycommon.job_chunking import JobChunker
from link_bot_pycommon.my_periodic_timer import MyPeriodicTimer
from link_bot_pycommon.pycommon import paths_to_json
from link_bot_pycommon.serialization import load_gzipped_pickle, dump_gzipped_pickle
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(None)


def cd_env_dist(aug_env,
                aug_res,
                aug_origin_point,
                data_env,
                data_res,
                data_origin_point):
    # chamfer distances
    subsample = 100
    aug_points = occupied_voxels_to_points(aug_env, aug_res, aug_origin_point)[::subsample]
    data_points = occupied_voxels_to_points(data_env, data_res, data_origin_point)[::subsample]
    chamfer_distance = evaluate(aug_points, data_points)
    return chamfer_distance


def compute_distance(aug_example: Dict, data_example: Dict):
    aug_rope = aug_example['rope']
    aug_rope_before = aug_rope[0]
    aug_rope_after = aug_rope[1]
    aug_joint_positions = aug_example['joint_positions']
    aug_joint_positions_before = aug_joint_positions[0]
    aug_joint_positions_after = aug_joint_positions[1]
    aug_env = aug_example['env']
    data_rope = data_example['rope']
    data_rope_before = data_rope[0]
    data_rope_after = data_rope[1]
    data_joint_positions = data_example['joint_positions']
    data_joint_positions_before = data_joint_positions[0]
    data_joint_positions_after = data_joint_positions[1]
    data_env = data_example['env']

    rope_before_dist = tf.linalg.norm(aug_rope_before - data_rope_before)
    rope_after_dist = tf.linalg.norm(aug_rope_after - data_rope_after)

    joint_positions_before_difference = aug_joint_positions_before - data_joint_positions_before
    joint_positions_before_difference_weighted = joint_positions_before_difference * joints_weights
    joint_positions_after_difference = aug_joint_positions_after - data_joint_positions_after
    joint_positions_after_difference_weighted = joint_positions_after_difference * joints_weights
    joint_positions_before_dist = tf.linalg.norm(joint_positions_before_difference_weighted)
    joint_positions_after_dist = tf.linalg.norm(joint_positions_after_difference_weighted)

    env_dist = cd_env_dist(aug_env, aug_example['res'], aug_example['origin_point'],
                           data_env, data_example['res'], data_example['origin_point'])

    distances = tf.stack([
        rope_before_dist,
        rope_after_dist,
        joint_positions_before_dist,
        joint_positions_after_dist,
        env_dist,
    ], axis=-1)
    return distances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('augdir', type=pathlib.Path)
    parser.add_argument('datadir', type=pathlib.Path)
    parser.add_argument('--regenerate', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    augfiles = list(args.augdir.glob("*.pkl.gz"))
    datafiles = list(args.datadir.glob("*.pkl.gz"))

    aug_name = '-'.join(args.augdir.name)
    data_name = '-'.join(args.datadir.name)
    name = f"{aug_name}-{data_name}"
    if args.debug:
        name = 'debug-' + name
    dirname = pathlib.Path('results') / name
    dirname.mkdir(exist_ok=True)
    logfilename = dirname / 'logfile.hjson'
    print(dirname)
    jc = JobChunker(logfilename)
    jc.store_result('augfiles', paths_to_json(augfiles))
    jc.store_result('datafiles', paths_to_json(datafiles))
    jc.store_result('weights', weights.tolist())

    timer = MyPeriodicTimer(10)  # save logfile every 10 seconds

    with Pool(2) as p:
        for i, augfile in enumerate(tqdm(augfiles)):
            if timer:
                jc.save()
            for j, datafile in enumerate(tqdm(datafiles, leave=False, position=1)):
                key = f"{i}-{j}"
                if jc.get_result(key) is None or args.regenerate:
                    aug_example = load_gzipped_pickle(augfile)
                    data_example = load_gzipped_pickle(datafile)
                    d = compute_distance(aug_example, data_example)
                    to_save = {
                        'distance':     d,
                        'aug_example':  aug_example,
                        'data_example': data_example,
                    }

                    # writing this is slow, so it's good to skip really far pairs
                    if tf.reduce_any(d > too_far):
                        continue

                    outfilename = dirname / f'{i}-{j}.pkl.gz'
                    dump_gzipped_pickle(to_save, outfilename)
                    jc.store_result(key, d.numpy().tolist(), save=False)
        jc.save()


if __name__ == '__main__':
    main()
