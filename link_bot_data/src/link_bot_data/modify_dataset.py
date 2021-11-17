#!/usr/bin/env python
import pathlib
from typing import Callable, Optional, Dict

import hjson
from colorama import Fore
from tqdm import tqdm

from arc_utilities import algorithms
from link_bot_data.base_dataset import BaseDatasetLoader
from link_bot_data.dataset_utils import write_example
from link_bot_data.new_base_dataset import NewBaseDatasetLoader
from moonshine.filepath_tools import load_params


def modify_hparams(in_dir: pathlib.Path, out_dir: pathlib.Path, update: Optional[Dict] = None):
    if update is None:
        update = {}
    out_dir.mkdir(exist_ok=True, parents=False)

    out_hparams = load_params(in_dir)
    algorithms.nested_dict_update(out_hparams, update)

    with (out_dir / 'hparams.hjson').open("w") as out_f:
        hjson.dump(out_hparams, out_f)


def modify_dataset(dataset_dir: pathlib.Path,
                   dataset,
                   outdir: pathlib.Path,
                   process_example: Callable,
                   save_format: str,
                   hparams_update: Optional[Dict] = None,
                   do_not_process: bool = True,
                   slow: bool = False):
    total_count = 0
    for full_output_directory, i, example in dataset_generator_all_modes(dataset_dir, dataset, outdir, hparams_update,
                                                                         do_not_process, slow):
        for out_example in process_example(dataset, example):
            write_example(full_output_directory, out_example, total_count, save_format)
            total_count += 1
    print(Fore.GREEN + f"Modified {total_count} examples")


def modify_dataset2(dataset_dir: pathlib.Path,
                    dataset,
                    outdir: pathlib.Path,
                    process_example: Callable,
                    save_format: str,
                    hparams_update: Optional[Dict] = None):
    total_count = 0
    for i, example in dataset_generator_all_modes2(dataset_dir, dataset, outdir, hparams_update):
        for out_example in process_example(dataset, example):
            write_example(outdir, out_example, total_count, save_format)
            total_count += 1
    print(Fore.GREEN + f"Modified {total_count} examples")


def filter_dataset(dataset_dir: pathlib.Path,
                   dataset: BaseDatasetLoader,
                   outdir: pathlib.Path,
                   should_keep: Callable,
                   save_format: str,
                   hparams_update: Optional[Dict] = None,
                   do_not_process: bool = True,
                   slow: bool = False,
                   start_at: int = 0):
    total_count = 0
    for full_output_directory, i, example in dataset_generator_all_modes(dataset_dir, dataset, outdir, hparams_update,
                                                                         do_not_process, slow):
        if i < start_at:
            continue
        if should_keep(dataset, example):
            total_count += 1
            if hasattr(dataset, 'scenario_metadata'):
                for k in dataset.scenario_metadata.keys():
                    example.pop(k)
            write_example(full_output_directory, example, total_count, save_format)
    print(Fore.GREEN + f"Kept {total_count} examples")


def dataset_generator_all_modes(dataset_dir: pathlib.Path,
                                dataset_loader,
                                outdir: pathlib.Path,
                                hparams_update: Optional[Dict] = None,
                                do_not_process: bool = True,
                                slow: bool = False):
    if hparams_update is None:
        hparams_update = {}

    modify_hparams(dataset_dir, outdir, hparams_update)

    for mode in ['train', 'test', 'val']:
        dataset = dataset_loader.get_datasets(mode=mode, shuffle=False, do_not_process=do_not_process, slow=slow)
        full_output_directory = outdir / mode
        full_output_directory.mkdir(parents=True, exist_ok=True)

        for i, example in enumerate(tqdm(dataset)):
            yield full_output_directory, i, example


def dataset_generator_all_modes2(dataset_dir: pathlib.Path,
                                 dataset: NewBaseDatasetLoader,
                                 outdir: pathlib.Path,
                                 hparams_update: Optional[Dict] = None):
    if hparams_update is None:
        hparams_update = {}

    modify_hparams(dataset_dir, outdir, hparams_update)

    for mode in ['train', 'test', 'val']:
        tf_dataset = dataset.get_datasets(mode=mode, shuffle=False).serial()
        for i, example in enumerate(tqdm(tf_dataset)):
            yield i, example
