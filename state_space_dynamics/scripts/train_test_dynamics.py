#!/usr/bin/env python
from time import time

from arc_utilities import ros_init
from moonshine.torch_runner import runner
from state_space_dynamics import train_test_dynamics

node_name = f"train_test_propnet_{int(time())}"


@ros_init.with_ros(node_name)
def main():
    runner(train_test_dynamics)


if __name__ == '__main__':
    main()
