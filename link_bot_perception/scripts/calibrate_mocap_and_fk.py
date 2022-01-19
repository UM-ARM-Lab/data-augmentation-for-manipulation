import numpy as np
import transformations

import rospy
from arc_utilities import ros_init
from arc_utilities.tf2wrapper import TF2Wrapper
from arc_utilities.transformation_helper import PoseToTransform
from arm_robots.hdt_michigan import Val
from ros_numpy import numpify


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
    calibrated_transforms = []
    mocap2base_markers_list = []
    for left_target, right_target in targets:
        mocap2base_markers = tf.get_transform('mocap_world', 'mocap_val_root_val_root')
        mocap2base_markers_list.append(mocap2base_markers)

        joint_state = val.get_state('both_arms')
        joint_angles = joint_state.joint_state.position
        joint_names = joint_state.joint_state.name

        mocap2left_tool = tf.get_transform('mocap_world', 'left_tool')
        base_fk_to_left_tool = numpify(
            PoseToTransform(val.jacobian_follower.fk(joint_angles, joint_names, 'left_tool').pose))

        mocap2right_tool = tf.get_transform('mocap_world', 'right_tool')
        base_fk_to_right_tool = numpify(
            PoseToTransform(val.jacobian_follower.fk(joint_angles, joint_names, 'right_tool').pose))

        mocap_to_base_fk_1 = mocap2left_tool @ transformations.inverse_matrix(base_fk_to_left_tool)
        base_markers_to_base_1 = np.linalg.solve(mocap2base_markers, mocap_to_base_fk_1)
        mocap_to_base_fk_2 = mocap2right_tool @ transformations.inverse_matrix(base_fk_to_right_tool)
        base_markers_to_base_2 = np.linalg.solve(mocap2base_markers, mocap_to_base_fk_2)



        calibrated_transforms.append(base_markers_to_base_1)
        calibrated_transforms.append(base_markers_to_base_2)

        # move to next pose
        val.follow_jacobian_to_position('both_arms', tool_names, [[left_target], [right_target]])
        rospy.sleep(5)

    # yes I know averaging transformation matrices is a bad idea but whatever it's fine
    mean_calibrated_transform = np.mean(calibrated_transforms, axis=0)
    p = np.concatenate((transformations.translation_from_matrix(mean_calibrated_transform), transformations.euler_from_matrix(mean_calibrated_transform)))
    print("Use this in mocap_to_val_static_transform.launch")
    print(p)

    val.disconnect()


if __name__ == '__main__':
    main()
