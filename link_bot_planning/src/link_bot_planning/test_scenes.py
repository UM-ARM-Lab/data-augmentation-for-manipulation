import pathlib
import pickle
import re
import shutil
from typing import Optional

import numpy as np

import rosbag
import rospy
from gazebo_msgs.msg import LinkStates
from sensor_msgs.msg import JointState


def make_scene_filename(root, idx):
    in_scene_file = root / f'scene_{idx:04d}.bag'
    return in_scene_file


def make_goal_filename(root, idx):
    in_goal_file = root / f'goal_{idx:04d}.pkl'
    return in_goal_file


def get_all_scenes(dirname: pathlib.Path):
    scenes = []
    bagfilenames = list(dirname.glob("*.bag"))
    for bagfilename in bagfilenames:
        m = re.match(r'scene_(\d+).bag', bagfilename.name)
        idx = int(m.group(1))
        scene = TestScene(dirname, idx)
        scenes.append(scene)
    return scenes


class TestScene:

    def __init__(self, root: pathlib.Path, idx: int):
        self.idx = idx
        self.root = root
        self.goal_filename = make_goal_filename(self.root, self.idx)
        self.scene_filename = make_scene_filename(self.root, self.idx)

        # read the data
        with rosbag.Bag(self.scene_filename) as bag:
            self.joint_state = next(iter(bag.read_messages(topics=['joint_state'])))[1]
            self.links_states = next(iter(bag.read_messages(topics=['links_states'])))[1]

        with self.goal_filename.open("rb") as goal_file:
            self.goal = pickle.load(goal_file)

    def save(self, force: Optional[bool] = False):
        save_test_scene_given_name(joint_state=self.joint_state,
                                   links_states=self.links_states,
                                   bagfile_name=self.scene_filename,
                                   force=force)
        with self.goal_filename.open("wb") as saved_goal_file:
            pickle.dump(self.goal, saved_goal_file)

    def change_index(self, new_idx: int, force: Optional[bool] = False):
        print(f'{self.idx}->{new_idx}')

        in_scene_file = make_scene_filename(self.root, self.idx)
        out_scene_file = make_scene_filename(self.root, new_idx)

        in_goal_file = make_goal_filename(self.root, self.idx)
        out_goal_file = make_goal_filename(self.root, new_idx)

        if not force:
            if out_scene_file.exists():
                raise RuntimeError(f"file {out_scene_file.as_posix()} exists!")
            if out_goal_file.exists():
                raise RuntimeError(f"file {out_goal_file.as_posix()} exists!")

        shutil.copy(in_scene_file, out_scene_file)
        shutil.copy(in_goal_file, out_goal_file)

        self.idx = new_idx

    def remove(self):
        self.goal_filename.unlink(missing_ok=False)
        self.scene_filename.unlink(missing_ok=False)
        print(f"Removed {self.goal_filename.as_posix()} and {self.scene_filename.as_posix()}")


def save_test_scene(joint_state: JointState,
                    links_states: LinkStates,
                    save_test_scenes_dir: pathlib.Path,
                    trial_idx: int,
                    force: bool):
    bagfile_name = make_scene_filename(save_test_scenes_dir, trial_idx)
    return save_test_scene_given_name(joint_state, links_states, bagfile_name, force)


def save_test_scene_given_name(joint_state: JointState,
                               links_states: LinkStates,
                               bagfile_name: pathlib.Path,
                               force: bool):
    if bagfile_name.exists() and not force:
        print(f"File {bagfile_name.as_posix()} already exists. Aborting")
        return None

    rospy.logdebug(f"Saving scene to {bagfile_name}")
    with rosbag.Bag(bagfile_name, 'w') as bag:
        bag.write('links_states', links_states)
        bag.write('joint_state', joint_state)

    return bagfile_name


def create_randomized_start_state(params, scenario, trial_idx):
    env_rng = np.random.RandomState(trial_idx)
    action_rng = np.random.RandomState(trial_idx)
    environment = scenario.get_environment(params)
    scenario.randomize_environment(env_rng, params)
    for i in range(10):
        state = scenario.get_state()
        action, invalid = scenario.sample_action(action_rng=action_rng,
                                                 environment=environment,
                                                 state=state,
                                                 action_params=params,
                                                 validate=True,
                                                 )
        scenario.execute_action(None, None, action)


def get_states_to_save():
    links_states: LinkStates = rospy.wait_for_message("gazebo/link_states", LinkStates)
    joint_state: JointState = rospy.wait_for_message("hdt_michigan/joint_states", JointState)
    return joint_state, links_states
