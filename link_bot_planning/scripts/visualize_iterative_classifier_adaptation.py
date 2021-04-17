#!/usr/bin/env python
import argparse
import logging
import pathlib
import re

import tensorflow as tf

from arc_utilities import ros_init
from link_bot_classifiers.train_test_classifier import eval_generator
from link_bot_data.dataset_utils import deserialize_scene_msg
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.job_chunking import JobChunker
from link_bot_pycommon.pycommon import pathify
from link_bot_pycommon.serialization import load_gzipped_pickle, dump_gzipped_pickle
from merrrt_visualization.rviz_animation_controller import RvizSimpleStepper
from moonshine.filepath_tools import load_hjson
from moonshine.moonshine_utils import remove_batch, numpify


def generate_iterative_classifier_outputs(ift_dir: pathlib.Path, regenerate: bool):
    classifier_datasets_dir = ift_dir / 'classifier_datasets'
    classifiers_dir = ift_dir / 'classifier_training_logdir'

    log = load_hjson(ift_dir / 'logfile.hjson')
    assert len(log['checkpoints']) == 1
    pretrained_classifier_dir = log['checkpoints'][0]
    classifiers_dirs = {
        0: pathlib.Path(pretrained_classifier_dir).parent
    }
    for iteration_classifier_dir in sorted(classifiers_dir.iterdir()):
        m = re.fullmatch(r"iteration_(\d+)_training_logdir", iteration_classifier_dir.name)
        classifier_iteration_index = int(m.group(1)) + 1
        classifiers_dirs[classifier_iteration_index] = iteration_classifier_dir

    root = ift_dir / 'eval'
    root.mkdir(exist_ok=True)

    logfile_name = root / 'logfile.hjson'
    job_chunker = JobChunker(logfile_name=logfile_name)

    for iteration_dataset_dir in sorted(classifier_datasets_dir.iterdir()):
        m = re.fullmatch(r"iteration_(\d+)_dataset", iteration_dataset_dir.name)
        dataset_iteration_idx = int(m.group(1))

        dataset_iteration_chunker = job_chunker.sub_chunker(str(dataset_iteration_idx))

        for classifier_iteration_index, iteration_classifier_dir in classifiers_dirs.items():

            data_filename = pathify(dataset_iteration_chunker.get_result(str(classifier_iteration_index)))
            if data_filename is None or regenerate:
                checkpoint = next(iteration_classifier_dir.iterdir()) / 'best_checkpoint'
                data_for_classifier_on_dataset = eval_generator(dataset_dirs=[iteration_dataset_dir],
                                                                checkpoint=checkpoint,
                                                                mode='all',
                                                                balance=False,
                                                                batch_size=1,
                                                                use_gt_rope=True)
                data_no_batch = []
                for e, o in data_for_classifier_on_dataset:
                    deserialize_scene_msg(e)
                    data_no_batch.append((numpify(remove_batch(e)), numpify(remove_batch(o))))

                data_filename = root / f'{dataset_iteration_idx}_{classifier_iteration_index}_data.pkl.gz'
                print(data_filename)
                dump_gzipped_pickle(data_no_batch, data_filename)
                dataset_iteration_chunker.store_result(str(classifier_iteration_index), data_filename)
            else:
                print(f"Found results {dataset_iteration_idx}, {classifier_iteration_index}")
                # from time import perf_counter
                # t0 = perf_counter()
                # data_no_batch = load_gzipped_pickle(data_filename)
                # print(perf_counter() - t0)

    # return data


def should_visualize(dataset_iteration_idx, classifier_iteration_idx, data):
    pass


def visualize(dataset_iteration_idx, classifier_iteration_idx, data):
    pass


def visualize_iterative_classifier_adaption(ift_dir: pathlib.Path, regenerate: bool):
    data = generate_iterative_classifier_outputs(ift_dir, regenerate)

    log = load_hjson(ift_dir / 'logfile.hjson')
    scenario = get_scenario(log['planner_params']['scenario'])

    rviz_stepper = RvizSimpleStepper()

    for i, data_i in data.items():
        for j, data_ij in data_i.items():
            if should_visualize(i, j, data_ij):
                visualize(i, j, data_ij)
                rviz_stepper.step()


@ros_init.with_ros("viz_itr")
def main():
    tf.get_logger().setLevel(logging.ERROR)
    parser = argparse.ArgumentParser()
    parser.add_argument("ift_dir", type=pathlib.Path)
    parser.add_argument("--regenerate", action='store_true')

    args = parser.parse_args()

    visualize_iterative_classifier_adaption(**vars(args))


if __name__ == '__main__':
    main()
