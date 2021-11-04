#!/usr/bin/env python
import argparse
import pathlib

from arc_utilities import ros_init
from learn_invariance.new_dynamics_dataset import NewDynamicsDatasetLoader
from link_bot_data.modify_dataset import modify_dataset2, modify_hparams
from link_bot_data.split_dataset import split_dataset_via_files


@ros_init.with_ros("add_vel")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')

    args = parser.parse_args()

    outdir = args.dataset_dir.parent / f"{args.dataset_dir.name}+vel"

    loader = NewDynamicsDatasetLoader([args.dataset_dir])
    s = loader.get_scenario()
    dataset = loader.get_datasets('all')
    e_for_getting_vel_state_keys = next(iter(dataset))
    _, vel_state_keys = s.propnet_add_vel(e_for_getting_vel_state_keys)
    s = loader.get_scenario()

    def _process_example(_, example):
        example, _ = s.propnet_add_vel(example)
        yield example

    modify_dataset2(dataset_dir=args.dataset_dir,
                    dataset=loader,
                    outdir=outdir,
                    process_example=_process_example,
                    hparams_update={},
                    save_format='pkl')
    split_dataset_via_files(args.dataset_dir, 'pkl')
    post_hparams_update = {
        'data_collection_params': {
            'state_keys': loader.state_keys + vel_state_keys,
        }
    }
    modify_hparams(outdir, outdir, post_hparams_update)


if __name__ == '__main__':
    main()
