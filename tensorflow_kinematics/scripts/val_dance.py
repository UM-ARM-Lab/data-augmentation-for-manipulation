import logging
import pathlib

import tensorflow as tf

import rospy
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.ros_pycommon import silence_urdfpy_warnings
from tensorflow_kinematics import hdt_ik
from tensorflow_kinematics.hdt_ik import HdtIK


def main():
    tf.get_logger().setLevel(logging.ERROR)
    rospy.init_node("ik_demo")

    silence_urdfpy_warnings()

    urdf_filename = pathlib.Path("/home/peter/catkin_ws/src/hdt_robot/hdt_michigan_description/urdf/hdt_michigan.urdf")
    scenario = get_scenario("dual_arm_rope_sim_val_with_robot_feasibility_checking")
    ik_solver = HdtIK(urdf_filename, scenario, num_restarts=1, max_iters=100, initial_value_noise=0)

    hdt_ik.logger.setLevel(logging.INFO)

    batch_size = 1

    gen = tf.random.Generator.from_seed(0)

    previous_solution = None
    for i in range(100):
        left_target_position = gen.uniform([batch_size, 3], [-0.4, 0, -0.2], [0.4, 1, 0.6])
        right_target_position = gen.uniform([batch_size, 3], [-0.4, 0, -0.2], [0.4, 1, 0.6])

        previous_solution, _ = ik_solver.solve(env_points=None,
                                               scene_msg=None,
                                               left_target_position=left_target_position,
                                               right_target_position=right_target_position,
                                               initial_value=previous_solution,
                                               viz=True)


if __name__ == '__main__':
    main()
