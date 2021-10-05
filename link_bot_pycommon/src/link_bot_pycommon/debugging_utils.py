import rospy


def debug_viz_batch_indices(batch_size):
    if rospy.get_param("SHOW_ALL", False):
        return range(batch_size)
    else:
        return [rospy.get_param("DEBUG_IDX", 0)]
