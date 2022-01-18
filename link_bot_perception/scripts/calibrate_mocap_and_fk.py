import numpy as np
import transformations

import ros_numpy
from arc_utilities import ros_init
from arc_utilities.tf2wrapper import TF2Wrapper
from arm_robots.hdt_michigan import Val


@ros_init.with_ros("calibrate_mocap_and_fk")
def main():
    np.set_printoptions(suppress=True, precision=6)
    val = Val(raise_on_failure=True)
    val._max_velocity_scale_factor = 1.0
    val.connect()

    tf = TF2Wrapper()

    tool_names = ['left_tool', 'right_tool']
    val.store_current_tool_orientations(tool_names)

    targets = [
        (np.array([0.85, 0.2, 0.9]), np.array([0.85, -0.2, 0.9])),
        (np.array([0.9, 0.2, 0.9]), np.array([0.9, -0.2, 0.9])),
        (np.array([0.85, 0.2, 0.85]), np.array([0.9, -0.2, 0.9])),
        (np.array([0.9, 0.2, 0.85]), np.array([0.9, -0.2, 0.855])),
        (np.array([0.9, 0.25, 0.85]), np.array([0.85, -0.25, 0.85])),
        (np.array([0.85, 0.25, 0.8]), np.array([0.85, -0.25, 0.8])),
        (np.array([0.85, 0.2, 0.75]), np.array([0.85, -0.25, 0.75])),
        (np.array([0.9, 0.2, 0.7]), np.array([0.9, -0.25, 0.7])),
    ]
    for left_target, right_target in targets:
        mocap2base_markers = tf.get_transform('mocap_world', 'mocap_val_root_val_root', time=rospy.Time.now())
        mocap2left_tool = tf.get_transform('mocap_world', 'left_tool', time=rospy.Time.now())
        base_fk_to_left_tool = ros_numpy.numpify(val.get_link_pose('left_tool'))

        mocap_to_base_fk = mocap2left_tool @ transformations.inverse_matrix(base_fk_to_left_tool)
        base_markers_to_base = np.linalg.solve(mocap2base_markers, mocap_to_base_fk)

        # move to next pose
        val.follow_jacobian_to_position('both_arms', tool_names, [[left_target], [right_target]])

    val.disconnect()


if __name__ == '__main__':
    main()
