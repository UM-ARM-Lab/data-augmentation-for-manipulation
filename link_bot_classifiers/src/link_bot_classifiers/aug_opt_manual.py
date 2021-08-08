import pathlib
from typing import Dict

import numpy as np
import tensorflow as tf

import rospy
import sdf_tools.utils_3d
from link_bot_classifiers.aug_opt_utils import debug_aug, debug_aug_sgd, transformation_obj_points
from link_bot_classifiers.aug_opt_v5 import opt_object_transform
from link_bot_pycommon.debugging_utils import debug_viz_batch_indices
from link_bot_pycommon.grid_utils import environment_to_vg_msg, send_voxelgrid_tf_origin_point_res, \
    subtract, binary_or, batch_point_to_idx
from merrrt_visualization.rviz_animation_controller import RvizSimpleStepper
from moonshine.filepath_tools import load_hjson
from moonshine.geometry import transformation_params_to_matrices, transform_points_3d, transformation_jacobian, \
    homogeneous
from moonshine.moonshine_utils import repeat, possibly_none_concat


def opt_object_manual(self,
                      inputs: Dict,
                      inputs_aug: Dict,
                      new_env: Dict,
                      obj_points,
                      object_points_occupancy,
                      res,
                      batch_size,
                      time):
    # viz new env
    if debug_aug():
        for b in debug_viz_batch_indices(batch_size):
            env_new_dict = {
                'env': new_env['env'].numpy(),
                'res': res[b].numpy(),
            }
            msg = environment_to_vg_msg(env_new_dict, frame='new_env_aug_vg', stamp=rospy.Time(0))
            self.debug.env_aug_pub1.publish(msg)

            send_voxelgrid_tf_origin_point_res(self.broadcaster,
                                               origin_point=new_env['origin_point'],
                                               res=res[b],
                                               frame='new_env_aug_vg')
            # stepper.step()

    manual_transforms_filename = pathlib.Path(
        "/media/shared/ift/v3-revert-debugging-1-1_1628263205_69ac9955d3/classifier_datasets/iteration_0000_dataset/manual_transforms.hjson")
    transformation_matrices = get_manual_transforms(inputs, manual_transforms_filename)
    obj_points_aug, to_local_frame = transformation_obj_points(obj_points, transformation_matrices)

    # this updates other representations of state/action that are fed into the network
    _, object_aug_update, local_origin_point_aug, local_center_aug = self.apply_object_augmentation_no_ik(
        transformation_matrices,
        to_local_frame,
        inputs,
        batch_size,
        time)
    inputs_aug.update(object_aug_update)

    if debug_aug_sgd():
        for b in debug_viz_batch_indices(batch_size):
            self.debug.send_position_transform(local_origin_point_aug[b], 'local_origin_point_aug')
            self.debug.send_position_transform(local_center_aug[b], 'local_center_aug')

    new_env_repeated = repeat(new_env, repetitions=batch_size, axis=0, new_axis=True)
    local_env_aug, _ = self.local_env_helper.get(local_center_aug, new_env_repeated, batch_size)
    local_env_aug_fix_deltas = tf.zeros([batch_size])
    self.local_env_aug_fix_delta = possibly_none_concat(self.local_env_aug_fix_delta, local_env_aug_fix_deltas, axis=0)

    return inputs_aug, local_origin_point_aug, local_center_aug, local_env_aug, local_env_aug_fix_deltas


def get_manual_transforms(inputs: Dict, manual_transforms_filename: pathlib.Path):
    manual_transforms = load_hjson(manual_transforms_filename)
    transformation_matrices = []
    for k in inputs['filename']:
        k_str = k.numpy().decode("utf-8")
        possible_transformation_matrices = manual_transforms[k_str]
        rand_idx = np.random.choice(range(len(possible_transformation_matrices)))
        transformation_matrix = possible_transformation_matrices[rand_idx]
        transformation_matrices.append(transformation_matrix)
    transformation_matrices = tf.constant(transformation_matrices, dtype=tf.float32)
    return transformation_matrices