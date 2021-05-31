import argparse
import pathlib
import pickle
from itertools import cycle

import hjson
import numpy as np
import transformations
from colorama import Fore
from progressbar import progressbar

import ros_numpy
from arc_utilities import ros_init
from gazebo_msgs.msg import LinkStates
from geometry_msgs.msg import Pose, Quaternion, Point
from link_bot_data import base_dataset
from link_bot_data.dataset_utils import pkl_write_example, data_directory, merge_hparams_dicts
from link_bot_gazebo.gazebo_services import GazeboServices
from link_bot_pycommon.get_scenario import get_scenario
from moonshine.filepath_tools import load_hjson
from std_msgs.msg import Float32

DEBUG_VIZ = True


def save_hparams(hparams, full_output_directory):
    hparams_filename = full_output_directory / 'params.hjson'
    with hparams_filename.open("w") as hparams_file:
        hjson.dump(hparams, hparams_file)


def load_mode_filenames(d: pathlib.Path, filenames_filename: pathlib.Path):
    with filenames_filename.open("r") as filenames_file:
        filenames = [l.strip("\n") for l in filenames_file.readlines()]
    return [d / p for p in filenames]


def get_filenames(dataset_dirs, mode: str):
    all_filenames = []
    for d in dataset_dirs:
        if mode == 'all':
            all_filenames.extend(load_mode_filenames(d, d / f'train.txt'))
            all_filenames.extend(load_mode_filenames(d, d / f'test.txt'))
            all_filenames.extend(load_mode_filenames(d, d / f'val.txt'))
        else:
            filenames_filename = d / f'{mode}.txt'
            all_filenames.extend(load_mode_filenames(d, filenames_filename))
    all_filenames = sorted(all_filenames)
    return all_filenames


class NewDynamicsDatasetLoader:

    def __init__(self, dataset_dirs, mode: str):
        self.dataset_dirs = dataset_dirs
        self.hparams = merge_hparams_dicts(dataset_dirs)
        self.mode = mode
        self.filenames = get_filenames(self.dataset_dirs, self.mode)

    def __iter__(self):
        for metadata_filename in self.filenames:
            metadata = load_hjson(metadata_filename)
            data_filename = metadata.pop("data")
            full_data_filename = metadata_filename.parent / data_filename
            with full_data_filename.open("rb") as example_file:
                example = pickle.load(example_file)
            example.update(metadata)
            yield example

    def __len__(self):
        return len(self.filenames)


def _homo(x):
    return np.concatenate([x, [1]])


def transform_link_states(m: np.ndarray, link_states: LinkStates):
    link_states_aug = LinkStates()
    for name, pose, twist in zip(link_states.name, link_states.pose, link_states.twist):
        translate = ros_numpy.numpify(pose.position)
        angles = transformations.euler_from_quaternion(ros_numpy.numpify(pose.orientation))
        link_transform = transformations.compose_matrix(angles=angles, translate=translate)
        link_transform_aug = m @ link_transform
        pose_aug = Pose()
        pose_aug.position = ros_numpy.msgify(Point, transformations.translation_from_matrix(link_transform_aug))
        pose_aug.orientation = ros_numpy.msgify(Quaternion, transformations.quaternion_from_matrix(link_transform_aug))
        twist_aug = twist  # ignoring this for now, it should be near zero already
        link_states_aug.name.append(name)
        link_states_aug.pose.append(pose_aug)
        link_states_aug.twist.append(twist_aug)
    return link_states_aug


def transform_dict_of_points_vectors(m: np.ndarray, d, keys):
    d_out = {}
    for k in keys:
        points = np.reshape(d[k], [-1, 3, 1])
        points_homo = np.concatenate([points, np.ones([points.shape[0], 1, 1])], axis=1)
        points_aug = np.matmul(m[None], points_homo)[:, :3, 0]
        d_out[k] = np.reshape(points_aug, -1)
    return d_out


def restore(gz, link_states, s):
    restore_action = {
        'left_gripper_position':  ros_numpy.numpify(
            link_states.pose[link_states.name.index('rope_3d::left_gripper')].position),
        'right_gripper_position': ros_numpy.numpify(
            link_states.pose[link_states.name.index('rope_3d::right_gripper')].position),
    }
    s.execute_action(None, None, restore_action, wait=False)
    gz.pause()
    gz.restore_from_link_states_msg(link_states, excluded_models=['kinect2', 'collision_sphere', 'my_ground_plane'])
    gz.play()


def sample_transform(rng, scaling):
    a = 0.25
    lower = np.array([-a, -a, -a, -np.pi, -np.pi, -np.pi])
    upper = np.array([a, a, a, np.pi, np.pi, np.pi])
    transform = rng.uniform(lower, upper).astype(np.float32) * scaling
    return transform
    # return [0, 0, 0, 0, 0, np.pi / 4]


@ros_init.with_ros("learn_invariance")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=pathlib.Path)
    parser.add_argument('outdir', type=pathlib.Path)

    args = parser.parse_args()

    dataset = NewDynamicsDatasetLoader([args.dataset], 'all')

    full_output_directory = data_directory(args.outdir)

    full_output_directory.mkdir(exist_ok=True)
    print(Fore.GREEN + full_output_directory.as_posix() + Fore.RESET)

    hparams = {
        'made_from': args.dataset.as_posix(),
    }
    save_hparams(hparams, full_output_directory)

    gz = GazeboServices()
    gz.setup_env(verbose=0, real_time_rate=0, max_step_size=0.01)
    s = get_scenario("floating_rope")
    params = load_hjson(pathlib.Path("../link_bot_data/collect_dynamics_params/floating_rope.hjson"))
    s.on_before_get_state_or_execute_action()
    rng = np.random.RandomState(0)
    action_rng = np.random.RandomState(0)

    n_output_examples = 100000
    infinite_dataset = cycle(dataset)
    scaling = 1e-6
    for example_idx in progressbar(range(n_output_examples), widgets=base_dataset.widgets):
        example = next(infinite_dataset)
        link_states_before = example['link_states'][0]  # t=0 arbitrarily
        restore(gz, link_states_before, s)

        environment = s.get_environment(params)
        scaling = scaling * 1.01
        if example_idx % 100 == 0:
            print(scaling)
        transform = sample_transform(rng, scaling)  # uniformly sample an se3 transformation
        m = transformations.compose_matrix(angles=transform[3:], translate=transform[:3])

        state_before = s.get_state()
        action, _ = s.sample_action(action_rng,
                                    environment,
                                    state_before,
                                    params,
                                    validate=True)  # sample an action

        s.execute_action(environment, state_before, action)
        state_after = s.get_state()
        link_states_after = state_after['link_states']

        # set the simulator to the augmented before state
        link_states_before_aug = transform_link_states(m, link_states_before)
        link_states_after_aug = transform_link_states(m, link_states_after)
        state_points_keys = ['rope', 'left_gripper', 'right_gripper']
        action_points_keys = ['left_gripper_position', 'right_gripper_position']
        state_before_aug = transform_dict_of_points_vectors(m, state_before, state_points_keys)
        state_before_aug['link_states'] = link_states_before_aug
        state_after_aug_expected = transform_dict_of_points_vectors(m, state_after, state_points_keys)
        state_after_aug_expected['link_states'] = link_states_after_aug
        action_aug = transform_dict_of_points_vectors(m, action, action_points_keys)

        # set the simulator state to make the augmented state
        restore(gz, link_states_before_aug, s)

        # execute the action and record the aug after state
        s.execute_action(environment, state_before_aug, action_aug)
        state_after_aug = s.get_state()

        if DEBUG_VIZ:
            s.plot_state_rviz(state_before, label='state_before', color='#ff0000')
            s.plot_action_rviz(state_before, action, label='action', color='#ff0000')
            s.plot_state_rviz(state_after, label='state_after', color='#aa0000')
            s.plot_state_rviz(state_before_aug, label='state_before_aug', color='#00ff00')
            s.plot_action_rviz(state_before_aug, action_aug, label='action_aug', color='#00ff00')
            s.plot_state_rviz(state_after_aug, label='state_after_aug', color='#00aa00')
            s.plot_state_rviz(state_after_aug_expected, label='state_after_aug_expected', color='#ffffff')
            error_viz = s.classifier_distance(state_after_aug, state_after_aug_expected)
            s.error_pub.publish(Float32(data=error_viz))

        out_example = {
            'state_before':             state_before,
            'state_after':              state_after,
            'action':                   action,
            'transformation':           transform,
            'state_before_aug':         state_before_aug,
            'state_after_aug':          state_after_aug,
            'state_after_aug_expected': state_after_aug_expected,
            'action_aug':               action_aug,
            'environment':              environment,
        }
        pkl_write_example(full_output_directory, out_example, example_idx)


if __name__ == '__main__':
    main()
