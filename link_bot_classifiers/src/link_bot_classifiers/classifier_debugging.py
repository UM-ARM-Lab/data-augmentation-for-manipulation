import numpy as np

from jsk_recognition_msgs.msg import BoundingBox
from link_bot_pycommon.grid_utils import vox_to_voxelgrid_stamped
from rospy import Publisher
from rviz_voxelgrid_visuals_msgs.msg import VoxelgridStamped


class ClassifierDebugging:
    def __init__(self):
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
