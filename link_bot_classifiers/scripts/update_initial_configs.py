#!/usr/bin/env python
import argparse
import pathlib
import pickle
import shutil

from tqdm import tqdm

import sdf_tools.utils_3d


def update_initial_configs(indir: pathlib.Path):
    for filename in tqdm(indir.glob("*.pkl")):
        bak = filename.parent / (filename.name + '.bak')
        shutil.copy(filename, bak)
        with filename.open('rb') as file:
            initial_config = pickle.load(file)
        env = initial_config['env']
        sdf, sdf_grad = sdf_tools.utils_3d.compute_sdf_and_gradient(env['env'],
                                                                    env['res'],
                                                                    env['origin_point'])
        if 'sdf' in initial_config:
            initial_config.pop('sdf')
        if 'sdf_grad' in initial_config:
            initial_config.pop('sdf_grad')
        initial_config['env']['sdf'] = sdf
        initial_config['env']['sdf_grad'] = sdf_grad
        with filename.open('wb') as file:
            pickle.dump(initial_config, file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('indir', type=pathlib.Path)

    args = parser.parse_args()

    update_initial_configs(**vars(args))


if __name__ == '__main__':
    main()
