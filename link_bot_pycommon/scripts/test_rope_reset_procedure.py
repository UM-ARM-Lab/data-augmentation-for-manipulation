#!/usr/bin/env python
import argparse
import pathlib

from arc_utilities import ros_init
from link_bot_pycommon.base_services import BaseServices
from link_bot_pycommon.get_scenario import get_scenario
from moonshine.filepath_tools import load_hjson


@ros_init.with_ros("test_rope_reset_procedure")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scenario')
    args = parser.parse_args()
    s = get_scenario(args.scenario)
    s.on_before_get_state_or_execute_action()

    params = load_hjson(pathlib.Path("../link_bot_planning/planner_configs/val_car/common.hjson"))

    services = BaseServices()
    bagfile_name = 'test_scenes/real_val_car2/scene_0000.bag'
    s.restore_from_bag(services, params, bagfile_name, force=True)

    s.robot.disconnect()


if __name__ == '__main__':
    main()
