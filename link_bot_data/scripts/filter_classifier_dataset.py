#!/usr/bin/env python
import numpy as np
import argparse
import inspect
import pathlib
from threading import Thread
from typing import Dict

import tensorflow as tf

import rospy
from arc_utilities import ros_init
from link_bot_data.classifier_dataset import ClassifierDatasetLoader
from link_bot_data.load_dataset import get_classifier_dataset_loader
from link_bot_data.modify_dataset import filter_dataset
from peter_msgs.msg import AnimationControl

pub = None


def keep_negative_hand_chosen(dataset_loader, example: Dict):
    global pub
    done_msg = AnimationControl()
    done_msg.command = AnimationControl.DONE

    negative = (example['is_close'][1] == 0)
    if not negative:
        return False

    midpoint = example['rope'].reshape([2,25,3])[0, 12]
    box_lower = np.array([-0.3, 0.55, -0.1])
    box_upper = np.array([0.3, 1.0, 0.0])
    near_hooks = np.all(box_lower < midpoint) and np.all(midpoint < box_upper)
    if not near_hooks:
        return False

    while True:
        def animate():
            dataset_loader.anim_transition_rviz(example)

        t = Thread(target=animate)
        t.start()

        rospy.sleep(1.0)  # force the user to wait and see the animation

        k = input("keep? [y/N]")
        pub.publish(done_msg)
        t.join()
        return k == 'y'


def keep_starts_close(dataset_loader, example: Dict):
    starts_close = (example['is_close'][0] == 1)
    return starts_close


def keep_starts_far(dataset_loader, example: Dict):
    starts_far = (example['is_far'][0] == 0)
    return starts_far


def keep_is_feasible(dataset: ClassifierDatasetLoader, example: Dict):
    joint_pos_dist = tf.linalg.norm(example['joint_positions'] - example['predicted/joint_positions'])
    feasible = joint_pos_dist < 0.075
    return feasible


# NOTE: dict of all free functions beginning with `keep_` that are available in the current scope
filter_funcs = {name: f for name, f in vars().items() if inspect.isfunction(f) and 'keep_' in f.__name__}


@ros_init.with_ros("modify_dynamics_dataset")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('filter_func_name', type=str, help='name of one of the above free functions for filtering')
    parser.add_argument('suffix', type=str, help='string added to the new dataset name')
    parser.add_argument('--start-at', type=int, help='start at', default=0)

    args = parser.parse_args()

    outdir = args.dataset_dir.parent / f"{args.dataset_dir.name}+{args.suffix}"

    hparams_update = {}

    if args.filter_func_name in filter_funcs:
        should_keep = filter_funcs[args.filter_func_name]
    else:
        print(f"No available function {args.filter_func_name}")
        print(f"Available functions are:")
        print(list(filter_funcs.keys()))
        return

    global pub
    pub = rospy.Publisher('/rviz_anim/control', AnimationControl, queue_size=10)

    dataset = get_classifier_dataset_loader([args.dataset_dir], load_true_states=True)
    filter_dataset(dataset_dir=args.dataset_dir,
                   dataset=dataset,
                   outdir=outdir,
                   save_format='pkl',
                   should_keep=should_keep,
                   hparams_update=hparams_update,
                   do_not_process=False,
                   start_at=args.start_at)


if __name__ == '__main__':
    main()
