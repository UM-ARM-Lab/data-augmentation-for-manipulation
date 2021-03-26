import pathlib
import shutil

from link_bot_planning.test_scenes import save_test_scene_given_name

import rosbag


def adjust_trash_pos(links_states, joint_state):
    idx = links_states.name.index("trash::body")
    links_states.pose[idx].position.x -= 0.01
    links_states.pose[idx].position.y -= 0.00
    # links_states.pose[idx].position.z += 0.00
    return links_states, joint_state

def replace_chair_with_trash(links_states, joint_state):
    idx = links_states.name.index("chair::link_0")
    links_states.name[idx] = "trash::body"
    links_states.pose[idx].position.x = 0.32
    links_states.pose[idx].position.y = 0.48
    links_states.pose[idx].position.z = -0.375
    links_states.pose[idx].orientation.x = 0
    links_states.pose[idx].orientation.y = 0
    links_states.pose[idx].orientation.z = 0
    links_states.pose[idx].orientation.w = 1
    return links_states, joint_state


def change_joint_config(links_states, joint_state):
    joint_state.position = list(joint_state.position)
    idx = joint_state.name.index("joint44")
    joint_state.position[idx] += 0.05
    return links_states, joint_state


def main():
    scenes_dir = pathlib.Path("/home/peter/catkin_ws/src/link_bot/link_bot_planning/test_scenes/party")
    scene_idx = 11
    stem = f'scene_{scene_idx:04d}'
    bagfilename = scenes_dir / f'{stem}.bag'
    backup_filename = scenes_dir / f'.{stem}.bag.bak'
    print(bagfilename)

    shutil.copy(bagfilename, backup_filename)

    with rosbag.Bag(bagfilename) as bag:
        joint_state = next(iter(bag.read_messages(topics=['joint_state'])))[1]
        links_states = next(iter(bag.read_messages(topics=['links_states'])))[1]

    new_links_states, new_joint_state = change_joint_config(links_states, joint_state)

    save_test_scene_given_name(new_joint_state, new_links_states, bagfilename, force=True)


if __name__ == '__main__':
    main()
