from typing import Dict, List

import numpy as np

from jsk_recognition_msgs.msg import BoundingBox
from link_bot_data.dataset_utils import add_predicted
from link_bot_pycommon.grid_utils import vox_to_voxelgrid_stamped
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from moonshine.moonshine_utils import to_list_of_strings, numpify
from moveit_msgs.msg import RobotTrajectory
from rospy import Publisher
from rviz_voxelgrid_visuals_msgs.msg import VoxelgridStamped
from trajectory_msgs.msg import JointTrajectoryPoint


class ClassifierDebugging:
    def __init__(self, scenario: ScenarioWithVisualization, state_keys: List[str], action_keys: List[str]):
        self.state_keys = state_keys
        self.action_keys = action_keys
        self.scenario = scenario
        self.raster_debug_pubs = [Publisher(f'raster_debug_{i}', VoxelgridStamped, queue_size=10) for i in range(5)]
        self.local_env_bbox_pub = Publisher('local_env_bbox', BoundingBox, queue_size=10)
        self.local_env_new_bbox_pub = Publisher('local_env_new_bbox', BoundingBox, queue_size=10, latch=True)
        self.aug_bbox_pub = Publisher('local_env_bbox_aug', BoundingBox, queue_size=10)
        self.env_aug_pub1 = Publisher("env_aug1", VoxelgridStamped, queue_size=10)
        self.env_aug_pub2 = Publisher("env_aug2", VoxelgridStamped, queue_size=10)
        self.env_aug_pub3 = Publisher("env_aug3", VoxelgridStamped, queue_size=10)
        self.env_aug_pub4 = Publisher("env_aug4", VoxelgridStamped, queue_size=10)
        self.env_aug_pub5 = Publisher("env_aug5", VoxelgridStamped, queue_size=10)
        self.object_state_pub = Publisher("object_state", VoxelgridStamped, queue_size=10)

    def clear(self):
        vg_empty = np.zeros((64, 64, 64))
        empty_msg = vox_to_voxelgrid_stamped(vg_empty, scale=0.01, frame='world')

        for p in self.raster_debug_pubs:
            p.publish(empty_msg)

        self.env_aug_pub1.publish(empty_msg)
        self.env_aug_pub2.publish(empty_msg)
        self.env_aug_pub3.publish(empty_msg)
        self.env_aug_pub4.publish(empty_msg)
        self.env_aug_pub5.publish(empty_msg)

    def plot_action_rviz(self, input_dict, b, label: str, color='red'):
        state_0 = numpify({k: input_dict[add_predicted(k)][b, 0] for k in self.state_keys})
        state_0['joint_names'] = input_dict['joint_names'][b, 0]
        action_0 = numpify({k: input_dict[k][b, 0] for k in self.action_keys})
        self.scenario.plot_action_rviz(state_0, action_0, idx=1, label=label, color=color)

        robot_state = {k: input_dict[k][b] for k in ['joint_names', add_predicted('joint_positions')]}
        display_traj_msg = make_robot_trajectory(robot_state)
        self.scenario.robot.display_robot_traj(display_traj_msg, label=label, color=color)

    def plot_state_rviz(self, input_dict, b, t, label: str, color='red'):
        state_t = numpify({k: input_dict[add_predicted(k)][b, t] for k in self.state_keys})
        state_t['joint_names'] = input_dict['joint_names'][b, t]
        self.scenario.plot_state_rviz(state_t, label=label, color=color)

        if 'is_close' in input_dict:
            self.scenario.plot_is_close(input_dict['is_close'][b, 1])
        else:
            self.scenario.plot_is_close(None)

        if 'error' in input_dict:
            error_t = input_dict['error'][b, 1]
            self.scenario.plot_error_rviz(error_t)
        else:
            self.scenario.plot_error_rviz(-999)

    def send_position_transform(self, p, child: str):
        self.scenario.tf.send_transform(p, [0, 0, 0, 1], 'world', child=child, is_static=False)


def make_robot_trajectory(robot_state: Dict):
    msg = RobotTrajectory()
    # use 0 because joint names will be the same at every time step anyways
    msg.joint_trajectory.joint_names = to_list_of_strings(robot_state['joint_names'][0])
    for i, position in enumerate(robot_state[add_predicted('joint_positions')]):
        point = JointTrajectoryPoint()
        point.positions = numpify(position)
        point.time_from_start.secs = i  # not really "time" but that's fine, it's just for visualization
        msg.joint_trajectory.points.append(point)
    return msg
