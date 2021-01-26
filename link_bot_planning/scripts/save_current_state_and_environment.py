#!/usr/bin/env python
import argparse
import json
import pathlib
import time

import colorama

from link_bot_gazebo_python.gazebo_services import GazeboServices
from link_bot_pycommon.floating_rope_scenario import FloatingRopeScenario
from link_bot_pycommon.ros_pycommon import get_environment_for_extents_3d
from link_bot_pycommon.serialization import dump_gzipped_pickle


def main():
    colorama.init(autoreset=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("recovery_dataset_dir", type=pathlib.Path,
                        help='the hparams.json file for the recovery dataset')

    args = parser.parse_args()
    scenario = FloatingRopeScenario()
    service_provider = GazeboServices()

    with args.recovery_dataset_dir.open("r") as hparams_file:
        hparams = json.load(hparams_file)
    data_collection_params = hparams['data_collection_params']

    data = []
    while True:
        state = scenario.get_state()
        environment = get_environment_for_extents_3d(extent=data_collection_params['extent'],
                                                     res=data_collection_params['res'],
                                                     service_provider=service_provider,
                                                     robot_name=scenario.robot_name())

        example = {
            'state': state,
            'environment': environment
        }

        data.append(example)

        k = input("state/environment saved. collect another? [y/N]")
        if k != 'y':
            break

    now = int(time.time())
    outfilename = f'state_and_environment_{now}.pkl.gz'
    print(f"writing to {outfilename}")
    dump_gzipped_pickle(data, outfilename)


if __name__ == "__main__":
    main()
