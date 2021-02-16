import pathlib

import numpy as np

import rosbag
import rospy
from gazebo_msgs.msg import LinkStates
from sensor_msgs.msg import JointState


def save_test_scene(joint_state: JointState,
                    links_states: LinkStates,
                    save_test_scenes_dir: pathlib.Path,
                    trial_idx: int):
    bagfile_name = save_test_scenes_dir / f'scene_{trial_idx:04d}.bag'
    return save_test_scene_given_name(joint_state, links_states, bagfile_name)


def save_test_scene_given_name(joint_state: JointState,
                               links_states: LinkStates,
                               bagfile_name: pathlib.Path):
    if bagfile_name.exists():
        rospy.logerr(f"File {bagfile_name.as_posix()} already exists. Aborting")
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
        action = scenario.sample_action(action_rng=action_rng,
                                        environment=environment,
                                        state=state,
                                        action_params=params,
                                        validate=True,
                                        )
        scenario.execute_action(action)


def get_states_to_save():
    links_states: LinkStates = rospy.wait_for_message("gazebo/link_states", LinkStates)
    joint_state: JointState = rospy.wait_for_message("hdt_michigan/joint_states", JointState)
    return joint_state, links_states
