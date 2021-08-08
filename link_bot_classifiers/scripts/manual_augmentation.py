#!/usr/bin/env python
import argparse
import pathlib
from copy import deepcopy

import hjson
import tensorflow as tf

import ros_numpy
import rospy
from arc_utilities import ros_init
from geometry_msgs.msg import Point, Pose
from link_bot_data.load_dataset import get_classifier_dataset_loader
from link_bot_data.visualization import plot_classifier_state_t
from link_bot_pycommon.basic_3d_pose_marker import Basic3DPoseInteractiveMarker
from link_bot_pycommon.matplotlib_utils import adjust_lightness
from moonshine.moonshine_utils import remove_batch
from peter_msgs.msg import AnimationControl
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker


done = False


def done_cb(msg: AnimationControl):
    global done

    if msg.command == AnimationControl.DONE:
        done = True


@ros_init.with_ros("manual_augmentation")
def main():
    global done

    parser = argparse.ArgumentParser()
    parser.add_argument("classifier_dataset_dir", type=pathlib.Path)
    parser.add_argument("--n", "-n", type=int, help='number of augmentations per example', default=2)

    args = parser.parse_args()

    done_sub = rospy.Subscriber('/rviz_anim/control', AnimationControl, done_cb)

    outfile = args.classifier_dataset_dir / 'manual_transforms.hjson'

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
        plot_classifier_state_t(s, keys, e, t=1, label=label + '_1', color=adjust_lightness(color, 0.8))

    manual_transforms = {}
    dataset = dataset_loader.get_datasets(mode='all')
    for example in dataset:
        example_filename = example['filename']

        original_rope_points = tf.reshape(example['rope'], [2, -1, 3])
        current_rope_points = original_rope_points
        rope_point = example['rope'][0, 0:3]

        initial_pose = Pose(position=ros_numpy.msgify(Point, rope_point))
        initial_pose.orientation.w = 1
        world_to_original = tf.convert_to_tensor(ros_numpy.numpify(initial_pose), tf.float32)
        im.set_pose(initial_pose)

        possible_transformation_matrices = []
        batch_size = 1
        for i in range(args.n):

            transformation_matrix = None
            done = False
            example_viz = deepcopy(example)
            while not done:
                to_local_frame = current_rope_points[0, 0][None, None]

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

                # viz(example, 'original', 'red')
                s.plot_environment_rviz(example)
                # viz(example_viz, 'aug', 'blue')

            possible_transformation_matrices.append(transformation_matrix.numpy())
        manual_transforms[example_filename] = possible_transformation_matrices

    with outfile.open("w") as file:
        hjson.dump(manual_transforms, file)


if __name__ == '__main__':
    main()
