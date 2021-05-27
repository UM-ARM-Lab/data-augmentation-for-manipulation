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


def replace_chair_with_trash(s: TestScene):
    idx = s.links_states.name.index("chair::link_0")
    s.links_states.name[idx] = "trash::body"
    s.links_states.pose[idx].position.x = 0.32
    s.links_states.pose[idx].position.y = 0.48
    s.links_states.pose[idx].position.z = -0.375
    s.links_states.pose[idx].orientation.x = 0
    s.links_states.pose[idx].orientation.y = 0
    s.links_states.pose[idx].orientation.z = 0
    s.links_states.pose[idx].orientation.w = 1
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
    #s.joint_state.position[idx] += delta)
    return s


def main():
    scene_dir = "long_hook2"
    scene_indices = range(0, 60)

    root_dir = pathlib.Path("/home/peter/catkin_ws/src/link_bot/link_bot_planning/test_scenes")
    scenes_fulldir = root_dir / scene_dir

    for scene_idx in scene_indices:
        s = TestScene(scenes_fulldir, scene_idx)

        s = shift_hook(s)
        s.save(force=True)


if __name__ == '__main__':
    main()
