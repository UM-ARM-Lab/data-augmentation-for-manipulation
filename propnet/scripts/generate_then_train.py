#!/usr/bin/env python
import argparse
import pathlib

from arc_utilities import ros_init
from augmentation.augment_dataset import augment_dynamics_dataset
from augmentation.load_aug_params import load_aug_params
from link_bot_pycommon.job_chunking import JobChunker
from propnet.magic import wand_lightning_magic
from propnet.train_test_propnet import train_main


@ros_init.with_ros("generate_then_train")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)
    parser.add_argument('outdir', type=pathlib.Path)
    parser.add_argument('aug_hparams', type=pathlib.Path)
    parser.add_argument('--model-hparams', type=pathlib.Path, default='hparams/cylinders.hjson')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--take', type=int)
    parser.add_argument('--n-augmentations', '-n', type=int, default=25)
    parser.add_argument('--train-m-models', '-m', type=int, default=3)
    parser.add_argument('--training-steps', '-t', type=int, default=125_000)

    args = parser.parse_args()

    args.outdir.mkdir(exist_ok=True, parents=True)

    aug_dataset_dir = args.outdir / f"{args.dataset_dir.name}+aug-{args.n_augmentations}"

    aug_hparams = load_aug_params(args.aug_hparams)
    aug_hparams['n_augmentations'] = args.n_augmentations

    c = JobChunker(logfile_name=args.outdir / 'logfile.hjson')

    if not c.has_result('aug_dataset_dir'):
        augment_dynamics_dataset(args.dataset_dir,
                                 mode='train',
                                 hparams=aug_hparams,
                                 take=args.take,  # for debugging
                                 outdir=aug_dataset_dir,
                                 visualize=args.visualize,
                                 n_augmentations=args.n_augmentations)
        c.store_result('aug_dataset_dir', aug_dataset_dir.as_posix())

    wand_lightning_magic()

    for model_seed in range(args.train_m_models):
        result_k = f'run_id-{model_seed}'
        run_id = c.get_result(result_k, None)
        if run_id is None:
            run_id = train_main(dataset_dir=aug_dataset_dir,
                                model_params=args.model_hparams,
                                batch_size=32,
                                epochs=-1,
                                seed=model_seed,
                                user='armlab',
                                steps=args.training_steps)
            c.store_result(result_k, run_id)

    print(c.log)


if __name__ == '__main__':
    main()
