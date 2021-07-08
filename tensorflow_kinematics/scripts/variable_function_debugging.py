import logging
import pathlib

import tensorflow as tf

import rospy
import urdf_parser_py
from link_bot_pycommon.get_scenario import get_scenario
from tensorflow_kinematics.hdt_ik import HdtIK

def main():
    tf.get_logger().setLevel(logging.DEBUG)


    def _on_error(_):
        pass


    rospy.init_node("debugging")
    urdf_parser_py.xml_reflection.core.on_error = _on_error

    urdf_filename = pathlib.Path("/home/peter/catkin_ws/src/hdt_robot/hdt_michigan_description/urdf/hdt_michigan.urdf")
    scenario = get_scenario("dual_arm_rope_sim_val_with_robot_feasibility_checking")
    ik_solver = HdtIK(urdf_filename, scenario, max_iters=10)
    initial_value = ik_solver.sample_joint_positions(10)
    q = tf.Variable(initial_value)


    def _slow():
        q = tf.Variable(initial_value)
        for _ in range(2):
            from time import perf_counter
            t0 = perf_counter()
            ik_solver.tree.fk_no_recursion(q)
            print(perf_counter() - t0)


    def _fast():
        q.assign(initial_value)
        for _ in range(2):
            from time import perf_counter
            t0 = perf_counter()
            ik_solver.tree.fk_no_recursion(q)
            print(perf_counter() - t0)


    _fast()
    print("---")
    _fast()

    print()
    print()

    _slow()
    print("---")
    _slow()


if __name__ == '__main__':
    main()