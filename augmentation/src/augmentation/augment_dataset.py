import pathlib
from copy import deepcopy
from typing import Dict, Callable, List, Optional

from colorama import Fore
from tqdm import tqdm

from augmentation.aug_opt import AugmentationOptimization
from learn_invariance.new_dynamics_dataset import NewDynamicsDatasetLoader
from link_bot_data.dataset_utils import write_example, add_predicted, index_to_filename2
from link_bot_data.local_env_helper import LocalEnvHelper
from link_bot_data.new_base_dataset import NewBaseDatasetLoader
from link_bot_data.new_classifier_dataset import NewClassifierDatasetLoader
from link_bot_data.split_dataset import split_dataset
from link_bot_data.visualization import classifier_transition_viz_t, DebuggingViz, init_viz_env, dynamics_viz_t
from link_bot_pycommon.debugging_utils import debug_viz_batch_indices
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from merrrt_visualization.rviz_animation_controller import RvizAnimation
from moonshine.indexing import try_index_batched_dict
from moonshine.moonshine_utils import remove_batch
from moonshine.numpify import numpify


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
                             take: Optional[int],
                             hparams: Dict,
                             outdir: pathlib.Path,
                             n_augmentations: int,
                             scenario=None,
                             visualize: bool = False,
                             batch_size: int = 32,
                             save_format='pkl'):
    dataset_loader = NewDynamicsDatasetLoader([dataset_dir])
    if scenario is None:
        scenario = dataset_loader.get_scenario()

    # current needed because mujoco IK requires a fully setup simulation...
    scenario.on_before_data_collection(dataset_loader.data_collection_params)

    def viz_f(_scenario, example, **kwargs):
        example = numpify(example)
        state_keys = list(filter(lambda k: k in example, dataset_loader.state_keys))
        anim = RvizAnimation(_scenario,
                             n_time_steps=example['time_idx'].size,
                             init_funcs=[
                                 init_viz_env
                             ],
                             t_funcs=[
                                 init_viz_env,
                                 dynamics_viz_t(metadata={},
                                                label='aug',
                                                state_metadata_keys=dataset_loader.state_metadata_keys,
                                                state_keys=state_keys,
                                                action_keys=dataset_loader.action_keys),
                             ])
        anim.play(example)

    debug_state_keys = dataset_loader.state_keys
    outdir = augment_dataset_from_loader(dataset_loader,
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
                                         save_format)

    return outdir


def augment_classifier_dataset(dataset_dir: pathlib.Path,
                               hparams: Dict,
                               outdir: pathlib.Path,
                               n_augmentations: int,
                               scenario,
                               mode: str = 'all',
                               take: Optional[int] = None,
                               visualize: bool = False,
                               batch_size: int = 128,
                               save_format='pkl'):
    dataset_loader = NewClassifierDatasetLoader([dataset_dir])
    viz_f = classifier_transition_viz_t(metadata={},
                                        state_metadata_keys=dataset_loader.state_metadata_keys,
                                        predicted_state_keys=dataset_loader.predicted_state_keys,
                                        true_state_keys=None)
    debug_state_keys = [add_predicted(k) for k in dataset_loader.state_keys]
    outdir = augment_dataset_from_loader(dataset_loader,
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
                                         save_format)
    split_dataset(outdir, val_split=0, test_split=0)
    return outdir


def augment_dataset_from_loader(dataset_loader: NewBaseDatasetLoader,
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
                                save_format='pkl'):
    aug = make_aug_opt(scenario, dataset_loader, hparams, debug_state_keys, batch_size)

    outdir.mkdir(exist_ok=True, parents=False)

    def augment(inputs):
        actual_batch_size = inputs['batch_size']
        if visualize:
            scenario.reset_viz()

            inputs_viz = remove_batch(inputs)
            viz_f(scenario, inputs_viz, idx=0, color='g')

        time = inputs['time_idx'].shape[1]

        for k in range(n_augmentations):
            output = aug.aug_opt(inputs, batch_size=actual_batch_size, time=time)
            output['augmented_from'] = inputs['full_filename']

            if visualize:
                for b in debug_viz_batch_indices(actual_batch_size):
                    output_b = try_index_batched_dict(output, b)
                    viz_f(scenario, output_b, idx=k, color='#0000ff88')

            yield output

    def out_examples_gen():
        for example in dataset:
            actual_batch_size = example['batch_size']
            out_example_keys = None

            for out_example in augment(example):
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

    dataset = dataset_loader.get_datasets(mode=mode).take(take)
    expected_total = (1 + n_augmentations) * len(dataset)
    dataset = dataset.batch(batch_size)

    print(Fore.GREEN + outdir.as_posix() + Fore.RESET)

    total_count = 0

    # copy in all of the data from modes we're not augmenting
    if mode != 'all':
        # += offsets example numbers so we don't overwrite the data we copy here with the augmentations
        total_count += copy_modes(dataset_loader, mode, outdir, save_format)

    need_to_write_hparams = True
    examples_names = []
    for out_example in tqdm(out_examples_gen(), total=expected_total):
        if 'sdf' in out_example:
            out_example.pop("sdf")
        if 'sdf_grad' in out_example:
            out_example.pop("sdf_grad")
        if need_to_write_hparams:
            scenario.aug_merge_hparams(dataset_dir, out_example, outdir)
            need_to_write_hparams = False
        write_example(outdir, out_example, total_count, save_format)
        example_name = index_to_filename2(total_count, save_format)
        examples_names.append(example_name)
        total_count += 1
    print(Fore.GREEN + outdir.as_posix() + Fore.RESET)

    if mode != 'all':
        with (outdir / f'{mode}.txt').open("w") as remaining_mode_f:
            remaining_mode_f.writelines([n + '\n' for n in examples_names])

    return outdir


def copy_modes(dataset_loader, mode, outdir, save_format):
    total_count = 0
    modes = ['train', 'val', 'test']
    modes.remove(mode)
    for remaining_mode in modes:
        remaining_mode_examples = []
        remaining_dataset = dataset_loader.get_datasets(mode=remaining_mode)
        for remaining_mode_example in tqdm(remaining_dataset):
            write_example(outdir, remaining_mode_example, total_count, save_format)
            example_name = index_to_filename2(total_count, save_format)
            remaining_mode_examples.append(example_name)
            total_count += 1

        with (outdir / f'{remaining_mode}.txt').open("w") as remaining_mode_f:
            remaining_mode_f.writelines([n + '\n' for n in remaining_mode_examples])
    return total_count


def make_aug_opt(scenario: ScenarioWithVisualization,
                 dataset_loader: NewBaseDatasetLoader,
                 hparams: Dict,
                 debug_state_keys: List[str],
                 batch_size: int):
    debug = DebuggingViz(scenario, debug_state_keys, dataset_loader.action_keys)
    local_env_helper = LocalEnvHelper(h=hparams['local_env_h_rows'],
                                      w=hparams['local_env_w_cols'],
                                      c=hparams['local_env_c_channels'])
    aug = AugmentationOptimization(scenario=scenario,
                                   debug=debug,
                                   local_env_helper=local_env_helper,
                                   hparams=hparams,
                                   batch_size=batch_size,
                                   state_keys=dataset_loader.state_keys,
                                   action_keys=dataset_loader.action_keys,
                                   points_state_keys=dataset_loader.points_state_keys,
                                   )
    return aug
