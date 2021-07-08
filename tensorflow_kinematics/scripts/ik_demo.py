import logging
import pathlib
from math import pi

import tensorflow as tf

import rospy
import urdf_parser_py.xml_reflection.core
from link_bot_pycommon.get_scenario import get_scenario
from merrrt_visualization.rviz_animation_controller import RvizSimpleStepper
from moonshine.simple_profiler import SimpleProfiler
from moonshine.tf_profiler_helper import TFProfilerHelper
from tensorflow_kinematics.hdt_ik import HdtIK, target


def main():
    tf.get_logger().setLevel(logging.DEBUG)
    rospy.init_node("ik_demo")

    def _on_error(_):
        pass

    urdf_parser_py.xml_reflection.core.on_error = _on_error

    urdf_filename = pathlib.Path("/home/peter/catkin_ws/src/hdt_robot/hdt_michigan_description/urdf/hdt_michigan.urdf")
    scenario = get_scenario("dual_arm_rope_sim_val_with_robot_feasibility_checking")
    ik_solver = HdtIK(urdf_filename, scenario)

    batch_size = 32
    viz = False
    profile = False

    gen = tf.random.Generator.from_seed(0)
    target_noise = gen.uniform([2, batch_size, 7],
                               [-1, -1, -1, 0, 0, 0, 0],
                               [1, 1, 1, 0, 0, 0, 0],
                               dtype=tf.float32) * 0.1
    left_target_pose = tf.tile(target(-0.2, 0.55, 0.2, -pi / 2, 0, 0), [batch_size, 1]) + target_noise[0]
    right_target_pose = tf.tile(target(0.2, 0.55, 0.22, -pi / 2 + 0.5, -pi, 0), [batch_size, 1]) + target_noise[1]
    # right_target_pose = tf.tile(target(0.0, 0.0, 0.0, 0, -pi, 0), [batch_size, 1])
    o = tf.constant([[[-0.25, 0.2, 0.2]]], tf.float32)
    env_points = tf.random.uniform([batch_size, 100, 3], -0.1, 0.1, dtype=tf.float32) + o

    # gen = tf.random.Generator.from_seed(0)
    # initial_noise = gen.uniform([batch_size, ik_solver.get_num_joints()], -1, 1, dtype=tf.float32) * 0.1
    # initial_value = tf.zeros([batch_size, ik_solver.get_num_joints()], dtype=tf.float32) + initial_noise

    logdir = "ik_demo_logdir"
    if profile:
        h = TFProfilerHelper(profile_arg=(1, 10), train_logdir=logdir)
    else:
        h = None

    total_p = SimpleProfiler()

    def _solve():
        ik_solver.solve(env_points=env_points,
                        left_target_pose=left_target_pose,
                        right_target_pose=right_target_pose,
                        viz=viz,
                        profiler_helper=h)

    total_p.profile(5, _solve, skip_first_n=1)
    print(total_p)

    q, converged = ik_solver.solve(env_points=env_points,
                                   left_target_pose=left_target_pose,
                                   right_target_pose=right_target_pose,
                                   viz=viz,
                                   profiler_helper=h)
    print(ik_solver.get_percentage_solved())
    stepper = RvizSimpleStepper()
    for b in range(batch_size):
        ik_solver.plot_robot_and_targets(q, left_target_pose, right_target_pose, b=b)
        stepper.step()


if __name__ == '__main__':
    main()
