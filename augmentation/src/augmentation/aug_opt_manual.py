import pathlib
from typing import Dict

import numpy as np
import tensorflow as tf

import rospy
from augmentation.aug_opt_utils import debug_aug, transform_obj_points
from link_bot_data.dataset_utils import add_predicted
from link_bot_pycommon.debugging_utils import debug_viz_batch_indices
from link_bot_pycommon.grid_utils import environment_to_vg_msg, send_voxelgrid_tf_origin_point_res
from moonshine.filepath_tools import load_hjson
from moonshine.moonshine_utils import repeat, possibly_none_concat

rng = np.random.RandomState(0)


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

    # manual_transforms_filename = pathlib.Path(
    #     "/media/shared/ift/v3-revert-debugging-1-1_1628263205_69ac9955d3/classifier_datasets/iteration_0000_dataset/manual_transforms.hjson")
    guess_filename = pathlib.Path(inputs['full_filename'][0].numpy().decode("utf-8"))
    guess_dataset_dir = guess_filename.parent
    manual_transforms_filename = guess_dataset_dir / 'manual_transforms.hjson'
    transformation_matrices = get_manual_transforms(inputs, manual_transforms_filename, rng)
    obj_points_aug, _ = transform_obj_points(obj_points, transformation_matrices)
    rope_points = tf.reshape(inputs[add_predicted('rope')], [2, -1, 3])
    to_local_frame = rope_points[0, 0][None, None]

    # this updates other representations of state/action that are fed into the network
    _, object_aug_update, local_origin_point_aug, local_center_aug = self.aug_apply_no_ik(
        transformation_matrices,
        to_local_frame,
        inputs,
        batch_size,
        time)
    inputs_aug.update(object_aug_update)

    if debug_aug():
        for b in debug_viz_batch_indices(batch_size):
            self.debug.send_position_transform(local_origin_point_aug[b], 'local_origin_point_aug')
            self.debug.send_position_transform(local_center_aug[b], 'local_center_aug')

    new_env_repeated = repeat(new_env, repetitions=batch_size, axis=0, new_axis=True)
    local_env_aug, _ = self.local_env_helper.get(local_center_aug, new_env_repeated, batch_size)
    local_env_aug_fix_deltas = tf.zeros([batch_size])
    self.local_env_aug_fix_delta = possibly_none_concat(self.local_env_aug_fix_delta, local_env_aug_fix_deltas, axis=0)

    return inputs_aug, local_origin_point_aug, local_center_aug, local_env_aug, local_env_aug_fix_deltas


def get_manual_transforms(inputs: Dict, manual_transforms_filename: pathlib.Path, rng):
    manual_transforms = load_hjson(manual_transforms_filename)
    transformation_matrices = []
    for k in inputs['filename']:
        k_str = k.numpy().decode("utf-8")
        possible_transformation_matrices = manual_transforms[k_str]
        no_transformation = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        possible_transformation_matrices.append(no_transformation)
        rand_idx = rng.choice(range(len(possible_transformation_matrices)))
        transformation_matrix = possible_transformation_matrices[rand_idx]
        transformation_matrices.append(transformation_matrix)
    transformation_matrices = tf.constant(transformation_matrices, dtype=tf.float32)
    return transformation_matrices
