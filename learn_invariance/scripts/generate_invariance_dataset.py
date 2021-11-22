#!/usr/bin/env python
import argparse
import pathlib

import hjson
import numpy as np
from colorama import Fore
from tqdm import trange, tqdm

from arc_utilities import ros_init
from link_bot_data.dataset_utils import pkl_write_example, make_unique_outdir
from link_bot_pycommon.get_scenario import get_scenario
from moonshine.filepath_tools import load_hjson


def save_hparams(hparams, full_output_directory):
    hparams_filename = full_output_directory / 'params.hjson'
    with hparams_filename.open("w") as hparams_file:
        hjson.dump(hparams, hparams_file)


def scaling_gen(n, scaling_type='linear'):
    scaling = 1e-6
    for j in range(n):
        if scaling_type == 'linear':
            scaling = j / n
            yield scaling
        else:
            scaling = scaling * 1.01
            if scaling >= 1:
                return


@ros_init.with_ros("generate_invariance_dataset")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_collection_params', type=pathlib.Path)
    parser.add_argument('outdir', type=pathlib.Path)
    parser.add_argument('--n-test-states', type=int, default=5)
    parser.add_argument('--visualize', action='store_true')

    args = parser.parse_args()

    full_output_directory = make_unique_outdir(args.outdir)

    full_output_directory.mkdir(exist_ok=True)
    print(Fore.GREEN + full_output_directory.as_posix() + Fore.RESET)

    transformation_dim = 6
    n_output_examples = int(np.sqrt(10 ** transformation_dim))
    params = load_hjson(args.data_collection_params)
    scenario_name = params['scenario']

    hparams = {
        'scenario':               scenario_name,
        'data_collection_params': params,
        'transformation_dim':     transformation_dim,
        'n_output_examples':      n_output_examples,
    }
    save_hparams(hparams, full_output_directory)

    s = get_scenario(scenario_name)
    s.on_before_data_collection(params)

    transform_sampling_rng = np.random.RandomState(0)
    state_rng = np.random.RandomState(0)
    action_rng = np.random.RandomState(0)

    example_idx = 0
    for i in trange(args.n_test_states, position=1):
        s.tinv_set_state(params, state_rng, args.visualize)
        example = s.tinv_generate_data(action_rng, params, args.visualize)

        for scaling in tqdm(scaling_gen(n_output_examples), position=2):
            print("FIXME:")
            scaling = 0.1
            # sample a transformation
            transform = s.tinv_sample_transform(transform_sampling_rng, scaling)  # uniformly sample a transformation

            example_aug_pred = s.tinv_apply_transformation(example, transform, args.visualize)

            # test out the augmentation
            s.tinv_set_state_from_aug_pred(example_aug_pred, args.visualize)
            example_aug_actual = s.tinv_generate_data_from_aug_pred(params, example_aug_pred, args.visualize)

            error = s.tinv_error(example_aug_actual, example_aug_pred)

            # if args.visualize:
            #     s.tinv_viz(example, label='', color='r')
            #     s.tinv_viz(example_aug, label='aug', color='b')

            out_example = {
                'transform': transform,
                'error':     error,
                'state_i':   i,
                'scaling':   scaling,
            }
            pkl_write_example(full_output_directory, out_example, example_idx)
            example_idx += 1


if __name__ == '__main__':
    main()
