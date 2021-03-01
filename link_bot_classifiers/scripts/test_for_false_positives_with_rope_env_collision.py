#!/usr/bin/env python
import argparse
import logging
import pathlib
from typing import Dict

import colorama
import numpy as np
import tensorflow as tf

import rospy
from arc_utilities import ros_init
from link_bot_classifiers import classifier_utils
from link_bot_classifiers.points_collision_checker import PointsCollisionChecker
from link_bot_data.classifier_dataset import ClassifierDatasetLoader
from link_bot_data.dataset_utils import batch_tf_dataset, remove_predicted
from link_bot_data.visualization import init_viz_env
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from link_bot_pycommon.serialization import dump_gzipped_pickle
from merrrt_visualization.rviz_animation_controller import RvizAnimation
from moonshine import filepath_tools
from moonshine.indexing import index_dict_of_batched_tensors_tf
from std_msgs.msg import Float32


@ros_init.with_ros("test_for_fp")
def main():
    colorama.init(autoreset=True)
    tf.get_logger().setLevel(logging.ERROR)

    np.set_printoptions(linewidth=250, precision=4, suppress=True)
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    parser.add_argument('checkpoint', type=pathlib.Path)
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'val', 'all'], default='test')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument('--use-gt-rope', action='store_true')
    parser.add_argument('--only-fp', action='store_true')
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--old-compat', action='store_true')
    parser.add_argument('--start-at', type=int, default=0)

    args = parser.parse_args()

    traj_idx_pub_ = rospy.Publisher("traj_idx_viz", Float32, queue_size=10)

    ###############
    # Model
    ###############
    trials_directory = pathlib.Path('trials').absolute()
    trial_path = args.checkpoint.parent.absolute()
    _, params = filepath_tools.create_or_load_trial(trial_path=trial_path,
                                                    trials_directory=trials_directory)

    ###############
    # Dataset
    ###############
    dataset = ClassifierDatasetLoader(args.dataset_dirs,
                                      load_true_states=True,
                                      use_gt_rope=args.use_gt_rope,
                                      threshold=args.threshold)
    tf_dataset = dataset.get_datasets(mode=args.mode)
    scenario = dataset.scenario

    ###############
    # Evaluate
    ###############
    tf_dataset = batch_tf_dataset(tf_dataset, args.batch_size, drop_remainder=True)

    model = classifier_utils.load_generic_model(args.checkpoint)

    cc = PointsCollisionChecker(pathlib.Path('trials/cc_baseline/none'), scenario)

    for batch_idx, example in enumerate(tf_dataset):

        if batch_idx < args.start_at:
            continue

        example.update(dataset.batch_metadata)
        predictions, _ = model.check_constraint_from_example(example, training=False)

        labels = tf.expand_dims(example['is_close'][:, 1:], axis=2)

        probabilities = predictions['probabilities']

        # Visualization
        example.pop("time")
        example.pop("batch_size")
        decisions = tf.squeeze(probabilities > 0.5, axis=-1)
        labels = tf.squeeze(tf.cast(labels, tf.bool), axis=-1)
        classifier_is_correct = tf.equal(decisions, labels)
        is_fp = tf.logical_and(tf.logical_not(labels), decisions)
        for b in range(args.batch_size):
            example_b = index_dict_of_batched_tensors_tf(example, b)

            example_b_pred = {}
            example_b_pred.update(example_b)
            example_b_pred.update({remove_predicted(k): example_b[k] for k in dataset.predicted_state_keys})

            rope_points_not_in_collision, _ = cc.check_constraint_from_example(example_b_pred)
            if rope_points_not_in_collision[0]:
                continue

            if args.only_fp:
                if not tf.reduce_all(is_fp[b]):
                    continue

            def _custom_viz_t(scenario: ScenarioWithVisualization, e: Dict, t: int):
                if t > 0:
                    accept_probability_t = predictions['probabilities'][b, t - 1, 0].numpy()
                else:
                    accept_probability_t = -999
                scenario.plot_accept_probability(accept_probability_t)

                traj_idx_msg = Float32()
                traj_idx_msg.data = batch_idx * args.batch_size + b
                traj_idx_pub_.publish(traj_idx_msg)

            anim = RvizAnimation(scenario=scenario,
                                 n_time_steps=dataset.horizon,
                                 init_funcs=[init_viz_env,
                                             dataset.init_viz_action(),
                                             ],
                                 t_funcs=[_custom_viz_t,
                                          dataset.classifier_transition_viz_t(),
                                          ExperimentScenario.plot_stdev_t,
                                          init_viz_env,
                                          ])

            dump_gzipped_pickle(example_b, pathlib.Path('debugging.pkl.gzip'))
            anim.play(example_b)


if __name__ == '__main__':
    main()
