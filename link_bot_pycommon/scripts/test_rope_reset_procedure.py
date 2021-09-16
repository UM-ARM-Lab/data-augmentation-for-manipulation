#!/usr/bin/env python
import pathlib

import numpy as np

from arc_utilities import ros_init
from link_bot_pycommon.get_scenario import get_scenario
from moonshine.filepath_tools import load_hjson


@ros_init.with_ros("test_rope_reset_procedure")
def main():
    s = get_scenario("real_val_with_robot_feasibility_checking")
    s.on_before_get_state_or_execute_action()
    env_rng = np.random.RandomState(0)

    params = load_hjson(pathlib.Path("../link_bot_planning/planner_configs/val_car/common.hjson"))

    s.randomize_environment(env_rng, params)

    s.robot.disconnect()


if __name__ == '__main__':
    main()
