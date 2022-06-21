import pathlib
from copy import deepcopy
from typing import Dict, Callable, List, Optional

from colorama import Fore
from tqdm import tqdm

from cylinders_simple_demo.aug_opt import AugmentationOptimization
from cylinders_simple_demo.cylinders_dynamics_dataset import CylindersDynamicsDatasetLoader
from cylinders_simple_demo.local_env_helper import LocalEnvHelper
from cylinders_simple_demo.cylinders_scenario import CylindersScenario
from cylinders_simple_demo.utils import empty_callable
from link_bot_data.tf_dataset_utils import pkl_write_example
from link_bot_data.visualization import DebuggingViz, init_viz_env, dynamics_viz_t
from link_bot_pycommon.debugging_utils import debug_viz_batch_indices
from merrrt_visualization.rviz_animation_controller import RvizAnimation
from moonshine.indexing import try_index_batched_dict
from moonshine.numpify import numpify
from moonshine.torch_and_tf_utils import remove_batch
from moonshine.torchify import torchify




def unbatch_examples(example, actual_batch_size):
    example_copy = deepcopy(example)
    if 'batch_size' in example_copy:
        example_copy.pop('batch_size')
    if 'time' in example_copy:
        example_copy.pop('time')

    # FIXME: this is a bad hack!
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
                             take: Optional[int] = None,
                             scenario=None,
                             visualize: bool = False,
                             batch_size: int = 32,
                             use_torch: bool = True):
    loader = CylindersDynamicsDatasetLoader([dataset_dir])
    if scenario is None:
        scenario = loader.get_scenario()

    # current needed because mujoco IK requires a fully setup simulation...
    scenario.on_before_data_collection(loader.data_collection_params)

    def viz_f(_scenario, example, **kwargs):
        example = numpify(example)
        state_keys = list(filter(lambda k: k in example, loader.state_keys))
        anim = RvizAnimation(_scenario,
                             n_time_steps=example['time_idx'].size,
                             init_funcs=[
                                 init_viz_env
                             ],
                             t_funcs=[
                                 init_viz_env,
                                 dynamics_viz_t(metadata={},
                                                label='aug',
                                                state_metadata_keys=loader.state_metadata_keys,
                                                state_keys=state_keys,
                                                action_keys=loader.action_keys),
                             ])
        anim.play(example)

    debug_state_keys = loader.state_keys
    outdir = augment_dataset_from_loader(loader,
                                         viz_f,
                                         dataset_dir,
                                         mode,
                                         take,
                                         hparams,
                                         outdir,
                                         n_augmentations,
                                         debug_state_keys,
                                         scenario,
                                         visualize,
                                         batch_size,
                                         use_torch)

    return outdir


def augment(scenario, aug, n_augmentations, inputs, visualize, viz_f, use_torch):
    actual_batch_size = inputs['batch_size']
    if visualize:
        scenario.reset_viz()

        inputs_viz = remove_batch(inputs)
        viz_f(scenario, inputs_viz, idx=0, color='g')

    time = inputs['time_idx'].shape[1]

    for k in range(n_augmentations):
        if use_torch:
            inputs = torchify(inputs)
        output = aug.aug_opt(inputs, batch_size=actual_batch_size, time=time)
        if use_torch:
            output = numpify(output)
        output['augmented_from'] = inputs['full_filename']

        if visualize:
            for b in debug_viz_batch_indices(actual_batch_size):
                output_b = try_index_batched_dict(output, b)
                viz_f(scenario, output_b, idx=k, color='#0000ff88')

        yield output


def out_examples_gen(scenario, aug, n_augmentations, dataset, visualize, viz_f, use_torch):
    for example in dataset:
        actual_batch_size = example['batch_size']
        out_example_keys = None

        for out_example in augment(scenario, aug, n_augmentations, example, visualize, viz_f, use_torch):
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


def augment_dataset_from_loader(loader: CylindersDynamicsDatasetLoader,
                                viz_f: Callable,
                                dataset_dir: pathlib.Path,
                                mode: str,
                                take: Optional[int],
                                hparams: Dict,
                                outdir: pathlib.Path,
                                n_augmentations: int,
                                debug_state_keys,
                                scenario,
                                visualize: bool = False,
                                batch_size: int = 128,
                                use_torch: bool = False):
    aug = make_aug_opt(scenario, loader, hparams, debug_state_keys, batch_size)

    outdir.mkdir(exist_ok=True, parents=False)
    dataset = loader.get_datasets(mode=mode).take(take)
    expected_total = (1 + n_augmentations) * len(dataset)
    dataset = dataset.batch(batch_size)

    print(Fore.GREEN + outdir.as_posix() + Fore.RESET)

    total_count = 0

    # copy in all of the data from modes we're not augmenting
    if mode != 'all':
        # += offsets example numbers so we don't overwrite the data we copy here with the augmentations
        total_count += copy_modes(loader, mode, outdir)

    need_to_write_hparams = True
    examples_names = []
    for out_example in tqdm(out_examples_gen(scenario, aug, n_augmentations, dataset, visualize, viz_f, use_torch),
                            total=expected_total):
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
            remaining_mode_f.writelines([n.as_posix() + '\n' for n in examples_names])

    return outdir


def copy_modes(loader, mode, outdir):
    total_count = 0
    modes = ['train', 'val', 'test']
    modes.remove(mode)
    for remaining_mode in modes:
        remaining_mode_examples = []
        remaining_dataset = loader.get_datasets(mode=remaining_mode)
        for remaining_mode_example in tqdm(remaining_dataset):
            _, full_metadata_filename = pkl_write_example(outdir, remaining_mode_example, total_count)
            remaining_mode_examples.append(full_metadata_filename)
            total_count += 1

        with (outdir / f'{remaining_mode}.txt').open("w") as remaining_mode_f:
            remaining_mode_f.writelines([n.as_posix() + '\n' for n in remaining_mode_examples])
    return total_count


def make_aug_opt(scenario: CylindersScenario,
                 loader: CylindersDynamicsDatasetLoader,
                 hparams: Dict,
                 debug_state_keys: List[str],
                 batch_size: int,
                 post_init_cb: Callable = empty_callable,
                 post_step_cb: Callable = empty_callable,
                 post_project_cb: Callable = empty_callable,
                 ):
    debug = DebuggingViz(scenario, debug_state_keys, loader.action_keys)
    local_env_helper = LocalEnvHelper(h=hparams['local_env_h_rows'],
                                      w=hparams['local_env_w_cols'],
                                      c=hparams['local_env_c_channels'])
    aug = AugmentationOptimization(scenario=scenario,
                                   debug=debug,
                                   local_env_helper=local_env_helper,
                                   hparams=hparams,
                                   batch_size=batch_size,
                                   state_keys=loader.state_keys,
                                   action_keys=loader.action_keys,
                                   post_init_cb=post_init_cb,
                                   post_step_cb=post_step_cb,
                                   post_project_cb=post_project_cb,
                                   )
    return aug
