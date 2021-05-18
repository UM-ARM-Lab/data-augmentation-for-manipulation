import tensorflow as tf

import rospy
from arc_utilities.ros_helpers import get_connected_publisher
from moonshine.raster_3d import points_to_voxel_grid
from rviz_voxelgrid_visuals.conversions import vox_to_voxelgrid_stamped
from rviz_voxelgrid_visuals_msgs.msg import VoxelgridStamped


def main():
    rospy.init_node("points_to_voxel_grid_demo")
    pub = get_connected_publisher('occupancy', VoxelgridStamped, queue_size=10)

    batch_indices = tf.constant([
        0,
        0,
        0,
        0,
        1,
        1,
        1,
    ], tf.int64)
    points = tf.constant([
        # 0
        [0, 0, 0],
        [0.1, 0, 0],
        [0, 0.1, 0],
        [0, 0, 0.1],
        # 1
        [0.3, 0.3, 0.3],
        [0.3, 0.2, 0.3],
        [0.3, 0.3, 0.2],
    ], tf.float32)
    res = tf.constant([
        # 0
        0.1,
        0.1,
        0.1,
        0.1,
        # 1
        0.1,
        0.1,
        0.1,
    ], tf.float32)
    origin = tf.constant([
        # 0
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        # 1
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ], tf.float32)
    batch_size = 2
    vg = points_to_voxel_grid(batch_indices, points, res, origin, h=4, w=4, c=4, batch_size=batch_size)

    for b in range(batch_size):
        pub.publish(vox_to_voxelgrid_stamped(vg[b].numpy(),
                                             scale=0.1,  # Each voxel is a 1cm cube
                                             frame_id='world',  # In frame "world", same as rviz fixed frame
                                             origin=origin.numpy()[b]))  # With origin centering the voxelgrid
        input("press enter")


if __name__ == '__main__':
    main()
