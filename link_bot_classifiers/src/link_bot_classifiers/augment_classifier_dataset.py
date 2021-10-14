import pathlib
from typing import Dict

from tqdm import tqdm

from link_bot_classifiers.nn_classifier import NNClassifier
from link_bot_data.dataset_utils import write_example
from link_bot_data.modify_dataset import modify_hparams
from link_bot_data.new_classifier_dataset import NewClassifierDatasetLoader
from link_bot_data.split_dataset import split_dataset
from link_bot_data.visualization import classifier_transition_viz_t
from moonshine.moonshine_utils import batch_examples_dicts, remove_batch


def augment_classifier_dataset(dataset_dir: pathlib.Path,
                               hparams: Dict,
                               outdir: pathlib.Path,
                               n_augmentations: int,
                               scenario,
                               visualize: bool = False,
                               save_format='pkl'):
    dataset_loader = NewClassifierDatasetLoader([dataset_dir])
    hparams['classifier_dataset_hparams'] = dataset_loader.hparams
    model = NNClassifier(hparams, batch_size=1, scenario=scenario)
    viz_f = classifier_transition_viz_t(metadata={},
                                        state_metadata_keys=dataset_loader.state_metadata_keys,
                                        predicted_state_keys=dataset_loader.predicted_state_keys,
                                        true_state_keys=None)

    def augment(inputs):
        inputs = batch_examples_dicts([inputs])
        if visualize:
            scenario.reset_planning_viz()

            inputs_viz = remove_batch(inputs)
            viz_f(scenario, inputs_viz, t=0, idx=0, color='g')
            viz_f(scenario, inputs_viz, t=1, idx=1, color='g')

        for k in range(n_augmentations):
            output = model.aug.aug_opt(inputs, batch_size=1, time=2)
            output = remove_batch(output)

            if visualize:
                viz_f(scenario, output, t=0, idx=2 * k + 2, color='#0000ff88')
                viz_f(scenario, output, t=1, idx=2 * k + 3, color='#0000ff88')

            yield output

    def out_examples_gen():
        for example in dataset:
            # the original example should also be included!
            yield example
            for out_example in augment(example):
                if 'error' in out_example:
                    out_example['metadata'] = {
                        'error': out_example['error'],
                    }
                yield out_example

    modify_hparams(dataset_dir, outdir, None)
    dataset = dataset_loader.get_datasets(mode='all', shuffle=False)
    # TODO: consider batching to speed things up?
    total_count = 0
    expected_total = (1 + n_augmentations) * len(dataset)
    for out_example in tqdm(out_examples_gen(), total=expected_total):
        write_example(outdir, out_example, total_count, save_format)
        total_count += 1
    split_dataset(outdir, val_split=0, test_split=0)

    return outdir
