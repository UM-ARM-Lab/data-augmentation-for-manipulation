#!/usr/bin/env python
import argparse
import pathlib
import shutil
from copy import deepcopy

import tensorflow as tf
from tqdm import tqdm

import ros_numpy
import rospy
from arc_utilities import ros_init
from geometry_msgs.msg import Point, Pose
from link_bot_data.load_dataset import get_classifier_dataset_loader
from link_bot_data.visualization import plot_classifier_state_t
from link_bot_pycommon.basic_3d_pose_marker import Basic3DPoseInteractiveMarker
from link_bot_pycommon.matplotlib_utils import adjust_lightness
from link_bot_pycommon.serialization import my_hdump
from moonshine.filepath_tools import load_hjson
from moonshine.indexing import index_state_action_with_metadata
from moonshine.moonshine_utils import remove_batch
from peter_msgs.msg import AnimationControl
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

done = False
skip_fwd = False
skip_bwd = False


def done_cb(msg: AnimationControl):
    global done
    global skip_fwd
    global skip_bwd

    if msg.command == AnimationControl.DONE:
        done = True
    if msg.command == AnimationControl.STEP_FORWARD:
        skip_fwd = True
    if msg.command == AnimationControl.STEP_BACKWARD:
        skip_bwd = True


@ros_init.with_ros("manual_augmentation")
def main():
    global done
    global skip_fwd
    global skip_bwd

    parser = argparse.ArgumentParser()
    parser.add_argument("classifier_dataset_dir", type=pathlib.Path)
    parser.add_argument("--n", "-n", type=int, help='number of augmentations per example', default=9)

    args = parser.parse_args()

    _ = rospy.Subscriber('/rviz_anim/control', AnimationControl, done_cb)

    outfile = args.classifier_dataset_dir / 'manual_transforms.hjson'
    outfile_backup = args.classifier_dataset_dir / 'manual_transforms.hjson.bak'
    if outfile.exists():
        shutil.copy(outfile, outfile_backup)

    def make_marker(scale: float):
        marker = Marker(type=Marker.SPHERE)
        r = 0.05
        marker.scale = Point(r, r, r)
        marker.color = ColorRGBA(0.5, 1.0, 0.5, 0.7)
        return marker

    im = Basic3DPoseInteractiveMarker(make_marker=make_marker)
    rospy.sleep(0.5)

    dataset_loader = get_classifier_dataset_loader(args.classifier_dataset_dir)
    s = dataset_loader.get_scenario()

    keys = dataset_loader.true_state_keys
    keys.remove("gt_rope")
    keys.remove("time_idx")

    def viz(e, label, color):
        plot_classifier_state_t(s, keys, e, t=0, label=label + '_0', color=color)
        s_for_a_t, a_t = index_state_action_with_metadata(e,
                                                          state_keys=dataset_loader.predicted_state_keys,
                                                          state_metadata_keys=dataset_loader.state_metadata_keys,
                                                          action_keys=dataset_loader.action_keys,
                                                          t=0)
        s.plot_action_rviz(s_for_a_t, a_t, label=label)
        plot_classifier_state_t(s, keys, e, t=1, label=label + '_1', color=adjust_lightness(color, 0.8))

    if outfile.exists():
        manual_transforms = load_hjson(outfile)
    else:
        manual_transforms = {}

    dataset = dataset_loader.get_datasets(mode='all')
    dataset = list(dataset)
    example_idx = 0
    while example_idx < len(dataset):
        example = dataset[example_idx]

        example_filename = example['filename']
        if example_filename not in manual_transforms:
            manual_transforms[example_filename] = []

        original_rope_points = tf.reshape(example['rope'], [2, -1, 3])
        rope_point = example['rope'][0, 0:3]

        initial_pose = Pose(position=ros_numpy.msgify(Point, rope_point))
        initial_pose.orientation.w = 1
        world_to_original = tf.convert_to_tensor(ros_numpy.numpify(initial_pose), tf.float32)
        im.set_pose(initial_pose)

        batch_size = 1
        for i in range(args.n):
            if len(manual_transforms[example_filename]) > i:
                print(f"Skipping {example_filename}, {i}")
                continue

            transformation_matrix = None
            done = False
            skip_fwd = False
            skip_bwd = False
            example_viz = deepcopy(example)
            while not done and not skip_fwd and not skip_bwd:
                to_local_frame = original_rope_points[0, 0][None, None]

                im_pose = im.get_pose()
                marker_to_world = tf.convert_to_tensor(ros_numpy.numpify(im_pose), tf.float32)
                transformation_matrix = tf.linalg.solve(world_to_original, marker_to_world)
                transformation_matrices = tf.expand_dims(transformation_matrix, axis=0)

                _, object_aug_update, _, _ = s.apply_object_augmentation_no_ik(transformation_matrices,
                                                                               to_local_frame,
                                                                               example,
                                                                               batch_size=batch_size,
                                                                               time=2,
                                                                               h=44,
                                                                               w=44,
                                                                               c=44)
                object_aug_update = remove_batch(object_aug_update)
                example_viz.update(object_aug_update)

                viz(example, 'original', 'red')
                s.plot_environment_rviz(example)
                viz(example_viz, 'aug', 'blue')
                rospy.sleep(0.05)

            if skip_fwd:
                print(f"skipping the rest of {example_filename}...")
                break
            elif skip_bwd:
                print(f"Going back...")
                example_idx -= 2
                break
            else:
                manual_transforms[example_filename].append(transformation_matrix.numpy())

            with outfile.open("w") as file:
                my_hdump(manual_transforms, file)
            print(f"saved {example_filename} {len(manual_transforms[example_filename])}/{args.n}")
        example_idx += 1


if __name__ == '__main__':
    main()
