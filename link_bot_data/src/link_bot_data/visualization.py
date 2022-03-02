from typing import Dict, Optional, List

import numpy as np
from matplotlib import colors

import rospy
from jsk_recognition_msgs.msg import BoundingBox
from link_bot_data.dataset_utils import add_predicted
from link_bot_pycommon.grid_utils_np import vox_to_voxelgrid_stamped
from link_bot_pycommon.matplotlib_utils import adjust_lightness
from link_bot_pycommon.pycommon import vector_to_points_2d
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from moonshine.indexing import index_time_with_metadata, index_state_action_with_metadata, try_index_time_with_metadata, \
    index_time_batched, index_time
from moonshine.numpify import numpify
from moonshine.tensorflow_utils import to_list_of_strings
from moonshine.torch_and_tf_utils import remove_batch
from moveit_msgs.msg import RobotTrajectory
from rospy import Publisher
from rviz_voxelgrid_visuals_msgs.msg import VoxelgridStamped
from std_msgs.msg import Float32
from trajectory_msgs.msg import JointTrajectoryPoint


class DebuggingViz:
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
        state_0 = {}
        for k in self.state_keys:
            if k in input_dict:
                state_0[k] = input_dict[k][b, 0]
        state_0 = numpify(state_0)
        state_0['joint_names'] = input_dict['joint_names'][b, 0]
        action_0 = numpify({k: input_dict[k][b, 0] for k in self.action_keys})
        self.scenario.plot_action_rviz(state_0, action_0, idx=1, label=label, color=color)

        if add_predicted('joint_positions') in input_dict:
            robot_state = {k: input_dict[k][b] for k in ['joint_names', add_predicted('joint_positions')]}
            display_traj_msg = make_robot_trajectory(robot_state)
            self.scenario.robot.display_robot_traj(display_traj_msg, label=label, color=color)

    def plot_state_rviz(self, input_dict, b, t, label: str, color='red'):
        plot_state_b_t(self.scenario, self.state_keys, input_dict, b=b, t=t, label=label, color=color)

    def send_position_transform(self, p, child: str):
        self.scenario.tf.send_transform(p, [0, 0, 0, 1], 'world', child=child, is_static=False)


def plot_rope_configuration(ax, rope_configuration, linewidth=None, linestyle=None, s=1, label=None, scatt=True,
                            **kwargs):
    xs, ys = vector_to_points_2d(rope_configuration)
    if scatt:
        ax.scatter(xs, ys, s=s, **kwargs)
    return ax.plot(xs, ys, linewidth=linewidth, label=label, linestyle=linestyle, **kwargs)


def plot_extents(ax, extent, linewidth=6, **kwargs):
    line = ax.plot([extent[0], extent[1], extent[1], extent[0], extent[0]],
                   [extent[2], extent[2], extent[3], extent[3], extent[2]],
                   linewidth=linewidth,
                   **kwargs)[0]
    return line


def dynamics_viz_t(metadata: Dict, state_metadata_keys, state_keys, action_keys, label='actual'):
    def _dynamics_transition_viz_t(scenario: ScenarioWithVisualization, example: Dict, t: int, **kwargs):
        weight = example.get('weight', 1)
        scenario.plot_weight_rviz(weight[t])

        label_extra = kwargs.pop("label", "")
        s_t = index_time_with_metadata(metadata, example, state_metadata_keys + state_keys, t=t)
        try_adding_aco(state=s_t, example=example)
        scenario.plot_state_rviz(s_t, label=label + label_extra, color='#ff0000ff')

        s_for_a_t, a_t = index_state_action_with_metadata(example,
                                                          state_keys=state_keys,
                                                          state_metadata_keys=state_metadata_keys,
                                                          action_keys=action_keys,
                                                          t=t)
        scenario.plot_action_rviz(s_for_a_t, a_t)

    return _dynamics_transition_viz_t


def recovery_transition_viz_t(metadata: Dict, state_keys: List[str]):
    def _recovery_transition_viz_t(scenario: ScenarioWithVisualization, example: Dict, t: int):
        e_t = index_time_with_metadata(metadata, example, state_keys, t=t)
        scenario.plot_state_rviz(e_t, label='', color='#ff0000ff', scale=1.1)

    return _recovery_transition_viz_t


def classifier_transition_viz_t(metadata: Dict, state_metadata_keys, predicted_state_keys, true_state_keys: Optional):
    def _classifier_transition_viz_t(scenario: ScenarioWithVisualization, example: Dict, t: int, **kwargs):
        label_extra = kwargs.pop("label", "")
        pred_t = try_index_time_with_metadata(metadata, example, state_metadata_keys + predicted_state_keys, t=t)
        try_adding_aco(state=pred_t, example=example)
        kw_color = kwargs.pop('color', None)
        pred_s_color = kw_color if kw_color is not None else '#0000ffff'
        scenario.plot_state_rviz(pred_t, label='predicted' + label_extra, color=pred_s_color, **kwargs)

        label_t = example['is_close'][t]
        scenario.plot_is_close(label_t)

        if true_state_keys is not None:
            true_t = try_index_time_with_metadata(metadata, example, state_metadata_keys + true_state_keys, t=t)
            try_adding_aco(state=true_t, example=example)
            true_s_color = kw_color if kw_color is not None else '#ff0000ff'
            scenario.plot_state_rviz(true_t, label='actual' + label_extra, scale=1.1, color=true_s_color, **kwargs)

        if add_predicted('accept_probability') in example:
            p_t = example[add_predicted('accept_probability')][t, 0]
            scenario.plot_accept_probability(p_t)

        if 'error' in example:
            scenario.plot_error_rviz(example['error'][t])

    return _classifier_transition_viz_t


def init_viz_action(metadata: Dict, action_keys, state_keys):
    def _init_viz_action(scenario: ScenarioWithVisualization, example: Dict):
        action = {k: example[k][0] for k in action_keys}
        pred_0 = index_time_with_metadata(metadata, example, state_keys, t=0)
        scenario.plot_action_rviz(pred_0, action)

    return _init_viz_action


def init_viz_env(scenario: ScenarioWithVisualization, example: Dict, t: Optional[int] = None):
    # the unused t arg makes it so we can pass this as either a t_func or a init_func
    scenario.plot_environment_rviz(example)


def stdev_viz_t(pub: rospy.Publisher):
    return float32_viz_t(pub, add_predicted('stdev'))


def recovery_probability_viz(pub: rospy.Publisher):
    return float32_viz(pub, 'recovery_probability')


def float32_viz(pub: rospy.Publisher, key: str):
    def _data_viz(scenario: ScenarioWithVisualization, example: Dict):
        data_msg = Float32()
        data_msg.data = example[key][0]
        pub.publish(data_msg)

    return _data_viz


def float32_viz_t(pub: rospy.Publisher, key: str):
    def _data_viz_t(scenario: ScenarioWithVisualization, example: Dict, t: int):
        data_t = example[key][t, 0]
        data_msg = Float32()
        data_msg.data = data_t
        pub.publish(data_msg)

    return _data_viz_t


def color_violinplot(parts, color):
    r, g, b, a = colors.to_rgba(color)
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        pc.set_alpha(a)
    for partname in ['cmeans', ]:
        if partname in parts:
            vp = parts[partname]
            vp.set_edgecolor('#dddddd')
            vp.set_alpha(a)
            vp.set_linewidth(3)
    for partname in ['cbars', 'cmins', 'cmaxes']:
        color_dark = adjust_lightness(color, 0.1)
        vp = parts[partname]
        vp.set_edgecolor(color_dark)
        vp.set_linewidth(1)
        vp.set_alpha(a)


def noise_x_like(y, nominal_x, noise=0.01):
    return np.random.normal(nominal_x, noise, size=y.shape[0])


def noisey_1d_scatter(ax, x, position, noise=0.01, **kwargs):
    ax.scatter(noise_x_like(x, position, noise), x, **kwargs)


def try_adding_aco(state: Dict, example: Dict):
    try:
        state['attached_collision_objects'] = example['scene_msg'].robot_state.attached_collision_objects
    except Exception:
        pass


def plot_state_b_t(scenario, state_keys, input_dict, b, t, label: str, color='red'):
    state_t = {}
    for k in state_keys:
        if k in input_dict:
            state_t[k] = input_dict[k][b, t]
    state_t = numpify(state_t)
    state_t['joint_names'] = input_dict['joint_names'][b, t]
    scenario.plot_state_rviz(state_t, label=label, color=color)

    if 'is_close' in input_dict:
        scenario.plot_is_close(input_dict['is_close'][b, 1])
    else:
        scenario.plot_is_close(None)

    if 'error' in input_dict:
        error_t = input_dict['error'][b, 1]
        scenario.plot_error_rviz(error_t)
    else:
        scenario.plot_error_rviz(-999)


def plot_state_t(scenario, state_keys, input_dict, t, label: str, color='red'):
    state_t = numpify({k: input_dict[add_predicted(k)][t] for k in state_keys})
    state_t['joint_names'] = input_dict['joint_names'][t]
    scenario.plot_state_rviz(state_t, label=label, color=color)

    if 'is_close' in input_dict:
        scenario.plot_is_close(input_dict['is_close'][1])
    else:
        scenario.plot_is_close(None)

    if 'error' in input_dict:
        error_t = input_dict['error'][1]
        scenario.plot_error_rviz(error_t)
    else:
        scenario.plot_error_rviz(-999)


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


def viz_pred_actual_t_batched(loader, model, example, outputs, s, t, threshold):
    actual_t = loader.index_time_batched(example, t)
    s.plot_state_rviz(actual_t, label='viz_actual', color='red')
    s.plot_action_rviz(actual_t, actual_t, color='gray', label='viz')
    model_state_keys = model.state_keys + model.state_metadata_keys
    prediction_t = numpify(remove_batch(index_time_batched(outputs, model_state_keys, t, False)))
    s.plot_state_rviz(prediction_t, label='viz_predicted', color='blue')
    error_t = s.classifier_distance(actual_t, prediction_t)
    s.plot_error_rviz(error_t)
    label_t = error_t < threshold
    s.plot_is_close(label_t)


def viz_pred_actual_t(loader, model, example, outputs, s, t, threshold):
    actual_t = loader.index_time(example, t)
    s.plot_state_rviz(actual_t, label='viz_actual', color='red')
    s.plot_action_rviz(actual_t, actual_t, color='gray', label='viz')
    model_state_keys = model.state_keys + model.state_metadata_keys
    prediction_t = numpify(index_time(outputs, model_state_keys, t, False))
    s.plot_state_rviz(prediction_t, label='viz_predicted', color='blue')
    error_t = s.classifier_distance(actual_t, prediction_t)
    s.plot_error_rviz(error_t)
    label_t = error_t < threshold
    s.plot_is_close(label_t)
