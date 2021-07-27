import argparse
import pathlib
import pickle

import numpy as np
from matplotlib import cm

import ros_numpy
import rospy
import sdf_tools.utils_3d
from arc_utilities import ros_init
from arc_utilities.tf2wrapper import TF2Wrapper
from geometry_msgs.msg import Point
from link_bot_data.rviz_arrow import rviz_arrow
from link_bot_data.visualization_common import make_delete_markerarray
from link_bot_pycommon import grid_utils
from link_bot_pycommon.basic_3d_pose_marker import Basic3DPoseInteractiveMarker
from link_bot_pycommon.grid_utils import batch_idx_to_point_3d_tf_res_origin_point, batch_point_to_idx, \
    vox_to_voxelgrid_stamped
from rviz_voxelgrid_visuals_msgs.msg import VoxelgridStamped
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker


def plot_points_rviz(pub, positions, colors, label: str, frame_id: str = 'vg', id: int = 0, **kwargs):
    scale = kwargs.get('scale', 0.02)

    msg = Marker()
    msg.header.frame_id = frame_id
    msg.header.stamp = rospy.Time.now()
    msg.ns = label
    msg.id = id
    msg.type = Marker.SPHERE_LIST
    msg.action = Marker.ADD
    msg.pose.orientation.w = 1
    msg.scale.x = scale
    msg.scale.y = scale
    msg.scale.z = scale
    msg.colors = [ColorRGBA(*c) for c in colors]
    for position in positions:
        p = Point(x=position[0], y=position[1], z=position[2])
        msg.points.append(p)

    pub.publish(msg)


@ros_init.with_ros("visualize_sdf")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('configs_dir', type=pathlib.Path)

    args = parser.parse_args()

    filename = list(args.configs_dir.glob("*.pkl"))[0]
    with filename.open('rb') as file:
        initial_config = pickle.load(file)
    env = initial_config['env']
    vg = env['env']
    res = env['res']
    origin_point = env['origin_point']

    # vg = np.zeros([10, 10, 10], dtype=np.float32)
    # vg[0:1, 0:10, 0:1] = 1
    # vg[0:10, 0:1, 0:1] = 1
    # vg[0:1, 0:1, 0:10] = 1

    sdf_grad = env['sdf_grad']
    sdf = env['sdf']
    # sdf, sdf_grad = sdf_tools.utils_3d.compute_sdf_and_gradient(vg, res, origin_point)

    arrows_pub = rospy.Publisher('arrows', MarkerArray, queue_size=10)
    env_viz_pub = rospy.Publisher('occupancy', VoxelgridStamped, queue_size=10)
    sdf_pub = rospy.Publisher('sdf_viz', Marker, queue_size=10)
    frame = 'env_vg'
    env_msg = vox_to_voxelgrid_stamped(vg, res, frame)
    tf = TF2Wrapper()

    t = 'im'

    if t == 'im':
        def make_marker(scale: float):
            query_point_radius = 0.02
            marker = Marker(type=Marker.SPHERE)
            marker.scale = Point(2 * query_point_radius, 2 * query_point_radius, 2 * query_point_radius)
            marker.color = ColorRGBA(0.5, 1.0, 0.5, 0.7)
            return marker

        query_point_im = Basic3DPoseInteractiveMarker(make_marker=make_marker, x=0.0, y=0.35, z=0.1)

    while not rospy.is_shutdown():
        if t == 'grid':
            grad_scale = 0.02
            grid_skip = 32
            indices = np.meshgrid(np.arange(sdf_grad.shape[0]), np.arange(sdf_grad.shape[1]),
                                  np.arange(sdf_grad.shape[2]))
            indices = np.stack(indices, axis=-1)
            points = batch_idx_to_point_3d_tf_res_origin_point(indices, res, origin_point)
        elif t == 'p':
            points = np.array([[0, 0.31, 0]], dtype=np.float32)
        elif t == 'im':
            grad_scale = 0.1
            grid_skip = 1
            pose = query_point_im.get_pose()
            point = ros_numpy.numpify(pose.position).astype(np.float32)
            points = np.expand_dims(point, axis=0)

        indices = batch_point_to_idx(points, res, origin_point).numpy()

        oob = np.any(indices < 0) or np.any(indices >= np.array(vg.shape)[None])
        if oob:
            print("oob!")
            continue

        indices = np.reshape(indices, [-1, 3])
        print(indices)
        rows, cols, channels = indices[:, 0], indices[:, 1], indices[:, 2]
        sdf_grad_at_points = sdf_grad[rows, cols, channels]
        sdf_at_points = sdf[rows, cols, channels]
        points_flat = points[::grid_skip]
        sdf_grad_flat = sdf_grad_at_points[::grid_skip] * grad_scale
        sdf_flat = sdf_at_points[::grid_skip]

        msg = MarkerArray()
        for i, (position, direction) in enumerate(zip(points_flat, sdf_grad_flat)):
            msg.markers.append(rviz_arrow(position,
                                          position + direction,
                                          frame_id='world',
                                          idx=i,
                                          label='viz_sdf'))
        arrows_pub.publish(make_delete_markerarray(0, 'viz_sdf'))
        arrows_pub.publish(msg)

        colors = cm.viridis(sdf_flat)
        plot_points_rviz(sdf_pub, points_flat, colors, label='sdf', frame_id=frame)

        env_viz_pub.publish(env_msg)
        grid_utils.send_voxelgrid_tf_origin_point_res(tf.tf_broadcaster, origin_point, res, frame=frame)
        tf.send_transform(origin_point, [0, 0, 0, 1], 'world', child='origin_point')


if __name__ == '__main__':
    main()
