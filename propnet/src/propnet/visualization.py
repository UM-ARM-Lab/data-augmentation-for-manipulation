import numpy as np

from moonshine.numpify import numpify


def plot_cylinders_paths(b, dataset, gt_pos, inputs, pred_pos, pred_vel, s):
    pred_dz = 0.01
    pred_alpha = 0.7
    line_scale = 0.004
    num_objs = inputs['num_objs'][b, 0, 0]
    for obj_idx in range(1, num_objs + 1):
        obj_pred_pos_xy = pred_pos[b, :, obj_idx].cpu().detach().numpy()
        obj_gt_pos_xy = gt_pos[b, :, obj_idx].cpu().detach().numpy()
        z = np.expand_dims(np.ones_like(obj_gt_pos_xy[..., 0], np.float32), -1) * 0.14
        obj_pred_pos_xyz = np.concatenate((obj_pred_pos_xy, z + pred_dz), -1)
        obj_gt_pos_xyz = np.concatenate((obj_gt_pos_xy, z), -1)
        gt_total_dist = np.linalg.norm(obj_gt_pos_xy[-1] - obj_gt_pos_xy[0])
        pred_total_dist = np.linalg.norm(obj_pred_pos_xy[-1] - obj_pred_pos_xy[0])
        max_total_dist = max(gt_total_dist, pred_total_dist)
        if max_total_dist > 0.005:
            s.plot_line_strip_rviz(obj_pred_pos_xyz, label=f'predicted{obj_idx}', color='#222277', scale=line_scale)
            s.plot_line_strip_rviz(obj_gt_pos_xyz, label=f'actual{obj_idx}', color='#772222', scale=line_scale)

    robot_gt_pos_xy = gt_pos[b, :, 0].cpu().detach().numpy()
    robot_gt_pos_xyz = np.concatenate((robot_gt_pos_xy, z), -1)
    s.plot_line_strip_rviz(robot_gt_pos_xyz, label='robot', color='#772277', scale=line_scale)
    state_t = {}
    for k in dataset.state_keys:
        if k in inputs:
            state_t[k] = numpify(inputs[k][b, -1])
    s.plot_state_rviz(state_t, label=f'actual', color=(1, 0, 0, 1))
    pred_state_t = s.propnet_outputs_to_state(inputs=inputs, pred_vel=pred_vel, pred_pos=pred_pos, b=b, t=-1,
                                              obj_dz=pred_dz)
    s.plot_state_rviz(pred_state_t, label=f'predicted', color=(0, 0, 1, pred_alpha), no_robot=True)
