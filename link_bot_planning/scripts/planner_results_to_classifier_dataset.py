#!/usr/bin/env python
import argparse
import pathlib
from typing import Dict

import colorama
import hjson
import tensorflow as tf

from link_bot_data.classifier_dataset_utils import generate_classifier_examples_from_batch
from link_bot_data.dataset_utils import tf_write_example
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.serialization import load_gzipped_pickle
from moonshine.moonshine_utils import sequence_of_dicts_to_dict_of_np_arrays


def plan_result_to_examples(result_idx: int, result: Dict, labeling_params: Dict):
    steps = result['steps']
    actual_path = result['steps']
    actions = result['actions']
    planned_path = result['planned_path']
    environment = result['environment']
    classifier_horizon = labeling_params['classifier_horizon']
    for prediction_start_t in range(0, len(actual_path) - classifier_horizon - 1, labeling_params['start_step']):
        inputs = {
            'traj_idx': tf.cast([result_idx] * classifier_horizon, tf.float32),
            'action':   actions[prediction_start_t:],
        }
        inputs.update(environment)
        outputs = sequence_of_dicts_to_dict_of_np_arrays(actual_path[prediction_start_t:])
        predictions = sequence_of_dicts_to_dict_of_np_arrays(planned_path[prediction_start_t:])
        yield from generate_classifier_examples_from_batch()


def main():
    colorama.init(autoreset=True)
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('results_dir', type=pathlib.Path)
    parser.add_argument('labeling_params', type=pathlib.Path)
    parser.add_argument('outdir', type=pathlib.Path)

    args = parser.parse_args()
    args.outdir.mkdir(exist_ok=True)

    compression_type = 'ZLIB'
    labeling_params = hjson.load(args.labeling_params.open("r"))

    # copy hparams from dynamics dataset into classifier dataset
    metadata_filename = args.results_dir / 'metadata.json'
    metadata = hjson.load(metadata_filename.open("r"))
    make_classifier_dataset_hparams(args.results_dir, metadata, labeling_params, args.outdir)

    example_idx = 0
    for trial_idx in metadata['trials']:
        results_filename = args.results_dir / f'{trial_idx}_metrics.pkl.gz'
        datum = load_gzipped_pickle(results_filename)
        for example_idx, out_example in enumerate(plan_result_to_examples(trial_idx, datum, labeling_params)):
            tf_write_example(args.out_dir, out_example, example_idx)
            example_idx += 1


def make_classifier_dataset_hparams(result_dir: pathlib.Path,
                                    metadata: Dict,
                                    labeling_params: Dict,
                                    outdir: pathlib.Path):
    planner_params = metadata['planner_params']
    fwd_model_dir = planner_params['fwd_model_dir'][0]
    fwd_model_hparams_filename = pathlib.Path(fwd_model_dir).parent / 'params.json'
    fwd_model_hparams = hjson.load(fwd_model_hparams_filename.open('r'))
    new_hparams_filename = outdir / 'hparams.hjson'
    classifier_dataset_hparams = {}
    classifier_dataset_hparams['dataset_dir'] = result_dir.as_posix()
    classifier_dataset_hparams['fwd_model_dir'] = fwd_model_dir
    for k, v in fwd_model_hparams['dynamics_dataset_hparams'].items():
        classifier_dataset_hparams[k] = v
    classifier_dataset_hparams['fwd_model_hparams'] = fwd_model_hparams
    classifier_dataset_hparams['labeling_params'] = labeling_params
    classifier_dataset_hparams['state_keys'] = [fwd_model_hparams['state_keys']]
    with new_hparams_filename.open("w") as new_hparams_file:
        hjson.dump(classifier_dataset_hparams, new_hparams_file, indent=2)
    return metadata


if __name__ == '__main__':
    main()
