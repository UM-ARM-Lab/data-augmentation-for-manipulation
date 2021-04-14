import argparse
import pathlib
import pickle

from arc_utilities import ros_init
from link_bot_planning.planning_evaluation import load_planner_params
from link_bot_pycommon.get_scenario import get_scenario


def save_initial_config(planner_params_filename: pathlib.Path, outdir: pathlib.Path, idx: int):
    scenario = get_scenario('dual_arm_rope_sim_val')

    filename = outdir / f'initial_config_{idx}.pkl'

    planner_params = load_planner_params(planner_params_filename)

    environment = scenario.get_environment(planner_params)
    state = scenario.get_state()

    initial_config = {
        'state': state,
        'env':   environment,
    }
    with filename.open('wb') as file:
        pickle.dump(initial_config, file)


@ros_init.with_ros("generate_pretransfer_dataset")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('planner_params_filename', type=pathlib.Path)
    parser.add_argument('outdir', type=pathlib.Path)
    parser.add_argument('idx', type=int)

    args = parser.parse_args()

    save_initial_config(**vars(args))


if __name__ == '__main__':
    main()
