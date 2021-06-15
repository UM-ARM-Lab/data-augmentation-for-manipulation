#!/usr/bin/env python
import argparse
import pathlib
from typing import Dict

from arc_utilities import ros_init
from link_bot_data.classifier_dataset import ClassifierDatasetLoader
from link_bot_data.load_dataset import guess_dataset_format
from link_bot_data.modify_dataset import modify_dataset, modify_dataset2
from link_bot_data.new_classifier_dataset import NewClassifierDatasetLoader
from link_bot_data.split_dataset import split_dataset


@ros_init.with_ros("modify_classifier_dataset")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('suffix', type=str, help='string added to the new dataset name')
    parser.add_argument('--save-format', type=str, choices=['pkl', 'tfrecord'], default='tfrecord')

    args = parser.parse_args()

    outdir = args.dataset_dir.parent / f"{args.dataset_dir.name}+{args.suffix}"

    def _process_example(dataset, example: Dict):
        # example['origin_point'] = extent_res_to_origin_point(example['extent'], example['res'])
        yield example

    hparams_update = {}

    dataset_format = guess_dataset_format(args.dataset_dir)
    if args.save_format is None:
        args.save_format = dataset_format

    if dataset_format == 'tfrecord':
        dataset = ClassifierDatasetLoader([args.dataset_dir], use_gt_rope=False, load_true_states=True)
        modify_dataset(dataset_dir=args.dataset_dir,
                       dataset=dataset,
                       outdir=outdir,
                       process_example=_process_example,
                       save_format=args.save_format,
                       hparams_update=hparams_update,
                       slow=False)
    else:
        dataset = NewClassifierDatasetLoader([args.dataset_dir])
        modify_dataset2(dataset_dir=args.dataset_dir,
                        dataset=dataset,
                        outdir=outdir,
                        process_example=_process_example,
                        hparams_update=hparams_update,
                        save_format=args.save_format)
        split_dataset(args.dataset_dir, 'pkl')


if __name__ == '__main__':
    main()
