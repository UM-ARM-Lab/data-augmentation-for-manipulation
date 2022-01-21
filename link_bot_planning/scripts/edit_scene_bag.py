#!/usr/bin/env python
import argparse
import pathlib

from link_bot_planning.test_scenes import TestScene


def move_poles_up(s: TestScene):
    idx = s.links_states.name.index("pole1::link_1")
    s.links_states.pose[idx].position.z += 0.75

    idx = s.links_states.name.index("pole2::link_1")
    s.links_states.pose[idx].position.z += 0.75

    return s


def adjust_link_pos(s: TestScene, link_name: str, dx: float = 0, dy: float = 0, dz: float = 0):
    idx = s.links_states.name.index(link_name)
    s.links_states.pose[idx].position.x += dx
    s.links_states.pose[idx].position.y += dy
    s.links_states.pose[idx].position.z += dz
    return s


def adjust_trash_pos(s: TestScene):
    return adjust_link_pos(s, "trash::body", 0.01, 0, 0)


def remove(s: TestScene):
    removes = [
        'left_wall::link',
        'right_wall::link',
        'front_wall::link',
        'back_wall::link',
    ]
    for n in removes :
        if n in s.links_states.name:
            s.links_states.name.remove(n)
            print("Removed ", n)
    return s


def rename(s: TestScene):
    renames = [
        ('car_engine3::body', 'car_engine2::body'),
        ('car_engine3::lift_tab1', 'car_engine2::lift_tab1'),
        ('car_engine3::lift_tab2', 'car_engine2::lift_tab2'),
        ('car_engine3::lift_tab3', 'car_engine2::lift_tab3'),
        ('car_engine3::lift_tab4', 'car_engine2::lift_tab4'),
    ]
    for old_name, new_name in renames:
        if old_name in s.links_states.name:
            idx = s.links_states.name.index(old_name)
            s.links_states.name[idx] = new_name
    return s


def change_joint_config(s: TestScene):
    s.joint_state.position = list(s.joint_state.position)
    idx = s.joint_state.name.index("joint44")
    s.joint_state.position[idx] += 0.05
    return s


def shift_hook(s: TestScene):
    print(s.links_states.name)
    idx = s.joint_state.name.index("long_hook1::link_1")
    delta = np.random.randn(3) * 0.1
    print(delta)
    # s.joint_state.position[idx] += delta)
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenes_dir", type=pathlib.Path)

    args = parser.parse_args()

    scene_indices = range(0, 1)


    for scene_idx in scene_indices:
        s = TestScene(args.scenes_dir, scene_idx)

        s = remove(s)
        s.save(force=True)


if __name__ == '__main__':
    main()
