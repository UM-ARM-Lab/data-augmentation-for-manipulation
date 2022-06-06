#!/usr/bin/env python
import argparse
import pathlib
import shutil
from multiprocessing import Pool

from tqdm import tqdm

from arc_utilities import ros_init
from link_bot_data.dataset_utils import modify_pad_env
from link_bot_data.tf_dataset_utils import pkl_write_example
from moonshine.my_torch_dataset import MyTorchDataset


def process_example(args):
    i, dataset, outdir = args
    example = dataset[i]

    modify_pad_env(example, 70, 50, 67)

    pkl_write_example(outdir, example, example['metadata']['example_idx'])


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

    for i in tqdm(range(len(dataset))):
        process_example((i, dataset, outdir))

    # with Pool() as pool:
    #     tasks = [(i, dataset, outdir) for i in range(len(dataset))]
    #     for _ in tqdm(pool.imap_unordered(process_example, tasks), total=len(tasks)):
    #         pass


if __name__ == '__main__':
    main()
