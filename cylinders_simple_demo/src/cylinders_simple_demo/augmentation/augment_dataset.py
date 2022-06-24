import pathlib
from copy import deepcopy
from typing import Dict, Optional

import torch
from colorama import Fore
from torch.utils.data import DataLoader
from tqdm import tqdm

from cylinders_simple_demo.augmentation.aug_opt import AugmentationOptimization
from cylinders_simple_demo.utils.cylinders_scenario import CylindersScenario
from cylinders_simple_demo.utils.data_utils import pkl_write_example, get_num_workers, my_collate
from cylinders_simple_demo.utils.my_torch_dataset import MyTorchDataset
from cylinders_simple_demo.utils.numpify import numpify
from cylinders_simple_demo.utils.torch_datasets_utils import dataset_take
from cylinders_simple_demo.utils.torchify import torchify


def unbatch_examples(example, actual_batch_size):
    example_copy = deepcopy(example)
    if 'batch_size' in example_copy:
        example_copy.pop('batch_size')
    if 'time' in example_copy:
        example_copy.pop('time')

    if 'metadata' in example_copy:
        example_copy.pop('metadata')
    for b in range(actual_batch_size):
        out_example_b = {k: v[b] for k, v in example_copy.items()}
        if 'error' in out_example_b:
            out_example_b['metadata'] = {
                'error': out_example_b['error'],
            }
        yield out_example_b


def augment_dynamics_dataset(dataset_dir: pathlib.Path,
                             mode: str,
                             hparams: Dict,
                             outdir: pathlib.Path,
                             n_augmentations: int,
                             batch_size: int,
                             take: Optional[int] = None):
    dataset = MyTorchDataset(dataset_dir, mode)
    dataset_taken = dataset_take(dataset, take)
    scenario = CylindersScenario()
    aug = make_aug_opt(scenario, dataset, hparams, batch_size)

    loader = DataLoader(dataset_taken, batch_size=batch_size, num_workers=get_num_workers(batch_size),
                        collate_fn=my_collate)

    outdir.mkdir(exist_ok=True, parents=False)
    expected_total = (1 + n_augmentations) * len(dataset_taken)

    print(Fore.GREEN + outdir.as_posix() + Fore.RESET)

    total_count = 0

    # # copy in all of the data from modes we're not augmenting
    # if mode != 'all':
    #     # += offsets example numbers so we don't overwrite the data we copy here with the augmentations
    #     total_count += copy_modes(dataset_dir, mode, outdir)

    need_to_write_hparams = True
    examples_names = []
    for out_example in tqdm(out_examples_gen(aug, n_augmentations, loader), total=expected_total):
        if 'sdf' in out_example:
            out_example.pop("sdf")
        if 'sdf_grad' in out_example:
            out_example.pop("sdf_grad")
        if need_to_write_hparams:
            scenario.aug_merge_hparams(dataset_dir, out_example, outdir)
            need_to_write_hparams = False
        _, full_metadata_filename = pkl_write_example(outdir, out_example, total_count)
        examples_names.append(full_metadata_filename)
        total_count += 1
    print(Fore.GREEN + outdir.as_posix() + Fore.RESET)

    if mode != 'all':
        with (outdir / f'{mode}.txt').open("w") as remaining_mode_f:
            remaining_mode_f.writelines([n.name + '\n' for n in examples_names])

    return outdir


def augment(aug, n_augmentations, inputs, device):
    actual_batch_size = len(inputs['filename'])

    time = inputs['time_idx'].shape[1]

    for k in range(n_augmentations):
        inputs = torchify(inputs)
        output = aug.aug_opt(inputs, batch_size=actual_batch_size, time=time, device=device)
        output = numpify(output)
        output['augmented_from'] = inputs['full_filename']

        yield output


def out_examples_gen(aug, n_augmentations, dataset):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    for example_cpu in dataset:
        example = {}
        for k, v in example_cpu.items():
            if isinstance(v, torch.Tensor):
                example[k] = v.to(device)
            else:
                example[k] = v
        actual_batch_size = len(example['filename'])
        out_example_keys = None

        for out_example in augment(aug, n_augmentations, example, device):
            if out_example_keys is None:
                out_example_keys = list(out_example.keys())
            yield from unbatch_examples(out_example, actual_batch_size)

        if 'batch_size' in out_example_keys:
            out_example_keys.remove('batch_size')

        # the original example should also be included!
        for original_example in unbatch_examples(example, actual_batch_size):
            # we lose some information when we augment, so only keep the keys that we have in the augmented data
            # for example, the image or the joint velocities.
            original_example_subset = {}
            for k in out_example_keys:
                if k in original_example:
                    original_example_subset[k] = original_example[k]
            yield original_example_subset


def copy_modes(dataset_dir, mode, outdir):
    total_count = 0
    modes = ['train', 'val', 'test']
    modes.remove(mode)
    for remaining_mode in modes:
        remaining_mode_examples = []
        remaining_dataset = MyTorchDataset(dataset_dir, mode=remaining_mode)
        for remaining_mode_example in tqdm(remaining_dataset):
            _, full_metadata_filename = pkl_write_example(outdir, remaining_mode_example, total_count)
            remaining_mode_examples.append(full_metadata_filename)
            total_count += 1

        with (outdir / f'{remaining_mode}.txt').open("w") as remaining_mode_f:
            remaining_mode_f.writelines([n.name + '\n' for n in remaining_mode_examples])
    return total_count


def make_aug_opt(scenario: CylindersScenario, dataset: MyTorchDataset, hparams: Dict, batch_size: int):
    aug = AugmentationOptimization(scenario=scenario, hparams=hparams, batch_size=batch_size,
                                   state_keys=dataset.params['data_collection_params']['state_keys'],
                                   action_keys=dataset.params['data_collection_params']['action_keys'])
    return aug
