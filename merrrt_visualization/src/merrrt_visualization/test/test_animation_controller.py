import rospy
from merrrt_visualization.rviz_animation_controller import RvizAnimationController, MultiRvizAnimationController


def simple_usage():
    rospy.init_node("test_animation_controller")

    r = RvizAnimationController(n_time_steps=10, ns='trajs')
    while not r.done:
        t = r.t()
        print(t)
        r.step()


def main():
    rospy.init_node("test_animation_controller")

    sub_anims = [
        RvizAnimationController(n_time_steps=5, ns='trajs'),
        RvizAnimationController(n_time_steps=10, ns='traj'),
    ]
    r = MultiRvizAnimationController(sub_anims)
    while not r.done:
        traj_idx, time_idx = r.t()
        print(traj_idx, time_idx)
        r.step()


if __name__ == '__main__':
    main()
