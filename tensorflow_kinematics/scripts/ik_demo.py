import logging
import pathlib
import pickle
from math import pi

import tensorflow as tf

import rospy
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.ros_pycommon import silence_urdfpy_warnings
from moonshine.simple_profiler import SimpleProfiler
from moonshine.tf_profiler_helper import TFProfilerHelper
from tensorflow_kinematics import hdt_ik
from tensorflow_kinematics.hdt_ik import HdtIK, target


def main():
    tf.get_logger().setLevel(logging.ERROR)
    rospy.init_node("ik_demo")

    silence_urdfpy_warnings()

    urdf_filename = pathlib.Path("/home/peter/catkin_ws/src/hdt_robot/hdt_michigan_description/urdf/hdt_michigan.urdf")
    scenario = get_scenario("dual_arm_rope_sim_val_with_robot_feasibility_checking")
    ik_solver = HdtIK(urdf_filename, scenario)

    hdt_ik.logger.setLevel(logging.INFO)

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

    # total_p = SimpleProfiler()
    #
    # def _solve():
    #     ik_solver.solve(env_points=env_points,
    #                     left_target_pose=left_target_pose,
    #                     right_target_pose=right_target_pose,
    #                     viz=viz,
    #                     profiler_helper=h)
    #
    # total_p.profile(5, _solve, skip_first_n=1)
    # print()
    # print(total_p)


    with pathlib.Path("/media/shared/pretransfer_initial_configs/car/initial_config_0.pkl").open("rb") as f:
        scene_msg = pickle.load(f)['env']['scene_msg']
    scene_msg_batched = [scene_msg] * batch_size

    scenario.planning_scene_viz_pub.publish(scene_msg)

    q, converged = ik_solver.solve(env_points=env_points,
                                   scene_msg=scene_msg_batched,
                                   left_target_pose=left_target_pose,
                                   right_target_pose=right_target_pose,
                                   viz=viz)
    print(ik_solver.get_percentage_solved())
    # from merrrt_visualization.rviz_animation_controller import RvizSimpleStepper
    # stepper = RvizSimpleStepper()
    # for b in range(batch_size):
    #     ik_solver.plot_robot_and_targets(q, left_target_pose, right_target_pose, b=b)
    #     stepper.step()


if __name__ == '__main__':
    main()
