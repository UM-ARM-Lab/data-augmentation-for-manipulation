#!/usr/bin/env python
import pathlib

from progressbar import progressbar

import rospy
from arc_utilities import ros_init
from learn_invariance.new_dynamics_dataset_loader import NewDynamicsDatasetLoader
from link_bot_data import base_dataset


@ros_init.with_ros("test_dataset_perf")
def main():
    dataset_loader = NewDynamicsDatasetLoader([pathlib.Path("invariance_data/rope_data2_1622483421_76df9d4e79/")])
    b = 32
    dataset = dataset_loader.get_dataset(mode='all').batch(batch_size=b).shuffle()

    from time import perf_counter
    t0 = perf_counter()
    for e in progressbar(dataset, widgets=base_dataset.widgets):
        if rospy.is_shutdown():
            return
        pass
    print(perf_counter() - t0)


if __name__ == '__main__':
    main()
