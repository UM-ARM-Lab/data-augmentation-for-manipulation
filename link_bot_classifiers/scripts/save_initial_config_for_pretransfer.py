#!/usr/bin/env python
import argparse
import pathlib
import pickle

from arc_utilities import ros_init
from link_bot_gazebo.gazebo_services import GazeboServices
from link_bot_planning.planning_evaluation import load_planner_params
from link_bot_planning.test_scenes import get_all_scenes, make_scene_filename
from link_bot_pycommon.get_scenario import get_scenario


def save_initial_config(planner_params_filename: pathlib.Path,
                        test_scenes_dir: pathlib.Path,
                        outdir: pathlib.Path):
    outdir.mkdir(exist_ok=True, parents=True)

    planner_params = load_planner_params(planner_params_filename)
    planner_params['res'] = 0.02

    scenario = get_scenario('dual_arm_rope_sim_val')
    scenario.on_before_get_state_or_execute_action()

    service_provider = GazeboServices()
    scenes = get_all_scenes(test_scenes_dir)
    for s in scenes:
        bagfile_name = make_scene_filename(test_scenes_dir, s.idx)
        scenario.restore_from_bag(service_provider, planner_params, bagfile_name)

        filename = outdir / f'initial_config_{s.idx}.pkl'

        environment = scenario.get_environment(planner_params)
        state = scenario.get_state()

        initial_config = {
            'state': state,
            'env':   environment,
        }
        with filename.open('wb') as file:
            pickle.dump(initial_config, file)

        print(f"Wrote {filename.as_posix()}")


@ros_init.with_ros("generate_pretransfer_dataset")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('planner_params_filename', type=pathlib.Path)
    parser.add_argument('test_scenes_dir', type=pathlib.Path)
    parser.add_argument('outdir', type=pathlib.Path)

    args = parser.parse_args()

    save_initial_config(**vars(args))


if __name__ == '__main__':
    main()
