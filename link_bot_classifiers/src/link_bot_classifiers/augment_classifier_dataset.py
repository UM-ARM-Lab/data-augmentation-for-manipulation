import pathlib
from copy import deepcopy
from typing import Dict

from tqdm import tqdm

from link_bot_classifiers.nn_classifier import NNClassifier
from link_bot_data.dataset_utils import write_example
from link_bot_data.modify_dataset import modify_hparams
from link_bot_data.new_classifier_dataset import NewClassifierDatasetLoader
from link_bot_data.split_dataset import split_dataset
from link_bot_data.visualization import classifier_transition_viz_t
from moonshine.moonshine_utils import remove_batch


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


def augment_classifier_dataset(dataset_dir: pathlib.Path,
                               hparams: Dict,
                               outdir: pathlib.Path,
                               n_augmentations: int,
                               scenario,
                               visualize: bool = False,
                               batch_size: int = 128,
                               save_format='pkl'):
    dataset_loader = NewClassifierDatasetLoader([dataset_dir])
    hparams['classifier_dataset_hparams'] = dataset_loader.hparams
    model = NNClassifier(hparams, batch_size=batch_size, scenario=scenario)
    viz_f = classifier_transition_viz_t(metadata={},
                                        state_metadata_keys=dataset_loader.state_metadata_keys,
                                        predicted_state_keys=dataset_loader.predicted_state_keys,
                                        true_state_keys=None)

    def augment(inputs):
        actual_batch_size = inputs['batch_size']
        if visualize:
            scenario.reset_planning_viz()

            inputs_viz = remove_batch(inputs)
            viz_f(scenario, inputs_viz, t=0, idx=0, color='g')
            viz_f(scenario, inputs_viz, t=1, idx=1, color='g')

        for k in range(n_augmentations):
            output = model.aug.aug_opt(inputs, batch_size=actual_batch_size, time=2)

            if visualize:
                viz_f(scenario, remove_batch(output), t=0, idx=2 * k + 2, color='#0000ff88')
                viz_f(scenario, remove_batch(output), t=1, idx=2 * k + 3, color='#0000ff88')

            yield output

    def out_examples_gen():
        for example in dataset:
            # the original example should also be included!
            actual_batch_size = example['batch_size']
            yield from unbatch_examples(example, actual_batch_size)
            for out_example in augment(example):
                yield from unbatch_examples(out_example, actual_batch_size)

    modify_hparams(dataset_dir, outdir, None)
    dataset = dataset_loader.get_datasets(mode='all', shuffle=False)
    expected_total = (1 + n_augmentations) * len(dataset)

    dataset = dataset.batch(batch_size)
    total_count = 0
    for out_example in tqdm(out_examples_gen(), total=expected_total):
        write_example(outdir, out_example, total_count, save_format)
        total_count += 1
    split_dataset(outdir, val_split=0, test_split=0)

    return outdir
