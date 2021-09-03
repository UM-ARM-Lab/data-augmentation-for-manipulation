#!/usr/bin/env python
import argparse
import logging
import pathlib

import rospkg
import tensorflow as tf
from colorama import Fore

from arc_utilities import ros_init
from link_bot_classifiers import train_test_classifier
from link_bot_data.base_collect_dynamics_data import PklDataCollector
from link_bot_data.classifier_dataset_utils import make_classifier_dataset_from_params_dict
from link_bot_data.split_dataset import split_dataset
from link_bot_gazebo.gazebo_services import get_gazebo_processes
from link_bot_planning.planning_evaluation import evaluate_multiple_planning, load_planner_params
from link_bot_planning.test_scenes import get_all_scene_indices
from link_bot_pycommon.job_chunking import JobChunker
from link_bot_pycommon.pycommon import pathify
from moonshine.filepath_tools import load_hjson

r = rospkg.RosPack()

fwd_model_dir = pathlib.Path("/media/shared/dy_trials/gt_rope_w_robot_ensemble")


def n(i):
    return f'iter_{i:04d}'


class RandomActionsBaseline:

    def __init__(self, logdir: pathlib.Path):
        self.logdir = logdir
        self.logfilename = logdir / 'logfile.hjson'

        self.job_chunker = JobChunker(self.logfilename)
        self.job_chunker.store_result('logfilename', self.logfilename.as_posix())

        self.collect_data_params_filename = self.job_chunker.load_or_prompt('collect_data_params_filename')
        self.n_trajs = int(self.job_chunker.load_or_prompt('n_trajs'))
        labeling_params_filename = pathlib.Path(self.job_chunker.load_or_prompt('labeling_params_filename'))
        self.labeling_params = load_hjson(labeling_params_filename)
        self.max_planning_trials = int(self.job_chunker.load_or_prompt('max_planning_trials'))
        self.planner_params_filename = pathlib.Path(self.job_chunker.load_or_prompt('planner_params_filename'))
        self.test_scenes_dir = pathlib.Path(self.job_chunker.load_or_prompt('test_scenes_dir'))

    def run(self):
        classifier_dataset_dirs = []
        n_iters = 5
        gazebo_processes = get_gazebo_processes()

        for i in range(n_iters):
            print(Fore.GREEN + f'Iteration {i}' + Fore.RESET)
            iter_chunker = self.job_chunker.sub_chunker(str(i))

            dynamics_dataset_dir = pathify(iter_chunker.get_result('dynamics_dataset_dir'))
            if dynamics_dataset_dir is None:
                [p.resume() for p in gazebo_processes]
                dynamics_dataset_dir = self.collect_dynamics_data(iter_chunker, i)
                [p.suspend() for p in gazebo_processes]
                print(dynamics_dataset_dir)
                iter_chunker.store_result('dynamics_dataset_dir', dynamics_dataset_dir.as_posix())

            new_classifier_dataset_dir = pathify(iter_chunker.get_result('new_classifier_dataset_dir'))
            if new_classifier_dataset_dir is None:
                [p.suspend() for p in gazebo_processes]
                new_classifier_dataset_dir = self.make_classifier_dataset(dynamics_dataset_dir, i)
                print(new_classifier_dataset_dir)
                iter_chunker.store_result('new_classifier_dataset_dir', new_classifier_dataset_dir.as_posix())

            classifier_dataset_dirs.append(new_classifier_dataset_dir)

            classifier_model_dir = pathify(iter_chunker.get_result('classifier_model_dir'))
            if classifier_model_dir is None:
                [p.suspend() for p in gazebo_processes]
                classifier_model_dir = self.learn_classifier(classifier_dataset_dirs, i)
                print(classifier_model_dir)
                iter_chunker.store_result('classifier_model_dir', classifier_model_dir.as_posix())

            planning_outdir = pathify(iter_chunker.get_result('planning_outdir'))
            if planning_outdir is None:
                [p.resume() for p in gazebo_processes]
                classifier_checkpoint = classifier_model_dir / 'best_checkpoint'
                planning_outdir = self.planning_evaluation(iter_chunker, classifier_checkpoint, i)
                [p.suspend() for p in gazebo_processes]
                print(planning_outdir)
                iter_chunker.store_result('planning_outdir', planning_outdir.as_posix())

    def get_collect_dynamics_data_params(self):
        collect_dynamics_data_params = load_hjson(pathlib.Path(self.collect_data_params_filename))
        return collect_dynamics_data_params

    def collect_dynamics_data(self, chunker: JobChunker, i: int):
        collect_dynamics_data_params = self.get_collect_dynamics_data_params()

        chunker.save()

        data_collector = PklDataCollector(params=collect_dynamics_data_params, seed=i, verbose=0)
        outdir = data_collector.collect_data(n_trajs=self.n_trajs,
                                             nickname='data',
                                             root=self.logdir / 'dynamics_datasets' / n(i))
        split_dataset(outdir)

        return outdir

    def make_classifier_dataset(self, dynamics_dataset_dir: pathlib.Path, i: int):
        outdir = self.logdir / 'classifier_datasets' / n(i)
        outdir.mkdir(parents=True, exist_ok=True)
        classifier_dataset_dir = make_classifier_dataset_from_params_dict(dataset_dir=dynamics_dataset_dir,
                                                                          fwd_model_dir=fwd_model_dir,
                                                                          labeling_params=self.labeling_params,
                                                                          outdir=outdir,
                                                                          batch_size=1,
                                                                          use_gt_rope=False,
                                                                          visualize=False,
                                                                          save_format='pkl')

        split_dataset(classifier_dataset_dir)

        return classifier_dataset_dir

    def learn_classifier(self, classifier_dataset_dirs, i: int):
        classifiers_module_path = pathlib.Path(r.get_path('link_bot_classifiers'))
        classifier_hparams = classifiers_module_path / 'hparams' / 'classifier' / 'nn_classifier2.hjson'
        trials_dir = self.logdir / 'classifiers'
        trial_path, final_val_metrics = train_test_classifier.train_main(dataset_dirs=classifier_dataset_dirs,
                                                                         model_hparams=classifier_hparams,
                                                                         log=n(i),
                                                                         trials_directory=trials_dir,
                                                                         checkpoint=None,
                                                                         batch_size=32,
                                                                         epochs=10,
                                                                         seed=i)
        return trial_path

    def planning_evaluation(self, chunker: JobChunker, classifier_model_dir: pathlib.Path, i: int):

        chunker.save()

        planner_params = load_planner_params(self.planner_params_filename)
        planner_params["classifier_model_dir"] = classifier_model_dir
        planner_params_tuples = [('random_actions_baseline', planner_params)]

        trials = list(get_all_scene_indices(self.test_scenes_dir))
        trials = trials[:self.max_planning_trials]

        outdir = self.logdir / 'planning_results' / n(i)
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
