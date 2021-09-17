#!/usr/bin/env python
import argparse
import logging
import pathlib
import shutil

import rospkg
import tensorflow as tf
from colorama import Fore
from tqdm import tqdm

from arc_utilities import ros_init
from link_bot_classifiers import train_test_classifier
from link_bot_data.load_dataset import get_classifier_dataset_loader
from link_bot_data.split_dataset import split_dataset
from link_bot_gazebo.gazebo_services import get_gazebo_processes
from link_bot_planning.planning_evaluation import evaluate_multiple_planning, load_planner_params
from link_bot_planning.test_scenes import get_all_scene_indices
from link_bot_pycommon.job_chunking import JobChunker
from link_bot_pycommon.pycommon import pathify

r = rospkg.RosPack()

fwd_model_dir = pathlib.Path("/media/shared/dy_trials/gt_rope_w_robot_ensemble")


def n(i, n_examples):
    return f'iteration_{i:04d}_examples_{n_examples:09d}'


def binary_search(n):
    q = [(0, n)]
    while len(q) > 0:
        low, high = q.pop(0)
        mid = (high + low) // 2
        yield mid
        if high - low > 1:
            q.append((low, mid - 1))
            q.append((mid + 1, high))
        elif high - low == 1:
            yield high


class RandomActionsBaseline:

    def __init__(self, logdir: pathlib.Path):
        self.logdir = logdir
        self.logfilename = logdir / 'logfile.hjson'

        self.job_chunker = JobChunker(self.logfilename)
        self.job_chunker.store_result('logfilename', self.logfilename.as_posix())

        self.full_classifier_dataset_dir = pathlib.Path(self.job_chunker.load_or_prompt('full_classifier_dataset_dir'))
        self.max_planning_trials = int(self.job_chunker.load_or_prompt('max_planning_trials'))
        self.planner_params_filename = pathlib.Path(self.job_chunker.load_or_prompt('planner_params_filename'))
        self.test_scenes_dir = pathlib.Path(self.job_chunker.load_or_prompt('test_scenes_dir'))

        self.full_classifier_dataset_loader = get_classifier_dataset_loader([self.full_classifier_dataset_dir])
        self.full_classifier_dataset = self.full_classifier_dataset_loader.get_datasets(mode='all')

    def run(self):
        gazebo_processes = get_gazebo_processes()

        full_dataset_n_examples = len(self.full_classifier_dataset)
        full_dataset_n_examples = 100

        resolution = 10
        n_batches = full_dataset_n_examples // resolution
        iterations = list(binary_search(n_batches))
        for i, j in enumerate(iterations):
            n_examples = resolution * j
            print(Fore.GREEN + f'Iteration {i}/{len(iterations)}, {n_examples} examples' + Fore.RESET)

            iter_chunker = self.job_chunker.sub_chunker(str(i))

            # take n_examples from the full dataset, make a copy of it
            # this is inefficient in terms of disc space but it makes the code/analysis easier
            classifier_dataset_i_dir = pathify(iter_chunker.get_result('classifier_dataset_i_dir'))
            if classifier_dataset_i_dir is None:
                classifier_dataset_i_dir = self.take_dataset(i, n_examples)
                print(classifier_dataset_i_dir)
                iter_chunker.store_result('classifier_dataset_i_dir', classifier_dataset_i_dir.as_posix())

            # train a classifier
            classifier_model_dir = pathify(iter_chunker.get_result('classifier_model_dir'))
            if classifier_model_dir is None:
                [p.suspend() for p in gazebo_processes]
                classifier_model_dir = self.learn_classifier(classifier_dataset_i_dir, i, n_examples)
                print(classifier_model_dir)
                iter_chunker.store_result('classifier_model_dir', classifier_model_dir.as_posix())

            # evaluate that classifier in planning
            planning_outdir = pathify(iter_chunker.get_result('planning_outdir'))
            if planning_outdir is None:
                [p.resume() for p in gazebo_processes]
                classifier_checkpoint = classifier_model_dir / 'best_checkpoint'
                planning_outdir = self.planning_evaluation(iter_chunker, classifier_checkpoint, i, n_examples)
                [p.suspend() for p in gazebo_processes]
                print(planning_outdir)
                iter_chunker.store_result('planning_outdir', planning_outdir.as_posix())

    def learn_classifier(self, classifier_dataset_dir, i: int, n_examples: int):
        classifiers_module_path = pathlib.Path(r.get_path('link_bot_classifiers'))
        classifier_hparams = classifiers_module_path / 'hparams' / 'classifier' / 'nn_classifier2.hjson'
        trials_dir = self.logdir / 'classifiers'
        trial_path, final_val_metrics = train_test_classifier.train_main(dataset_dirs=[classifier_dataset_dir],
                                                                         model_hparams=classifier_hparams,
                                                                         log=n(i, n_examples),
                                                                         trials_directory=trials_dir,
                                                                         checkpoint=None,
                                                                         batch_size=32,
                                                                         epochs=10,
                                                                         seed=i)
        return trial_path

    def planning_evaluation(self, chunker: JobChunker, classifier_model_dir: pathlib.Path, i: int, n_examples: int):

        chunker.save()

        planner_params = load_planner_params(self.planner_params_filename)
        planner_params["classifier_model_dir"] = [
            classifier_model_dir,
            '/media/shared/cl_trials/new_feasibility_baseline/none',
        ]

        planner_params_tuples = [('random_actions_baseline', planner_params)]

        trials = list(get_all_scene_indices(self.test_scenes_dir))
        trials = trials[:self.max_planning_trials]

        outdir = self.logdir / 'planning_results' / n(i, n_examples)
        planning_logfile_name = outdir / 'logfile.hjson'
        outdir = evaluate_multiple_planning(outdir=outdir,
                                            logfile_name=planning_logfile_name,
                                            planners_params=planner_params_tuples,
                                            trials=trials,
                                            test_scenes_dir=self.test_scenes_dir,
                                            verbose=0,
                                            how_to_handle='raise',
                                            )
        return outdir

    def take_dataset(self, i, n_examples):
        classifier_dataset_i = self.full_classifier_dataset.take(n_examples)
        classifier_dataset_i_dir = self.logdir / 'classifier_datasets' / f"iteration_{i:04d}_examples_{n_examples:09d}"
        classifier_dataset_i_dir.mkdir(parents=True, exist_ok=True)
        parent = classifier_dataset_i.filenames[0].parent

        def _copy(_filename):
            _out_filename = classifier_dataset_i_dir / _filename.name
            if not _out_filename.exists():
                shutil.copy(_filename.as_posix(), _out_filename.as_posix())

        _copy(parent / 'hparams.hjson')
        for filename in tqdm(classifier_dataset_i.filenames, 'copying dataset files...'):
            _copy(filename)
            gz_filename = filename.parent / (filename.name + '.gz')
            _copy(gz_filename)

        split_dataset(classifier_dataset_i_dir)

        return classifier_dataset_i_dir


@ros_init.with_ros("random_actions_baseline")
def main():
    tf.get_logger().setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", type=pathlib.Path)

    args = parser.parse_args()

    rab = RandomActionsBaseline(args.logdir)
    rab.run()


if __name__ == '__main__':
    main()
