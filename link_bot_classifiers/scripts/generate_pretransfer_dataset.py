import argparse
import itertools
import pathlib
import pickle
import warnings
from time import perf_counter
from typing import Dict

import numpy as np

from arc_utilities import ros_init
from link_bot_data.dataset_utils import tf_write_example
from link_bot_data.files_dataset import FilesDataset
from link_bot_planning.get_planner import get_planner
from link_bot_planning.my_planner import PlanningQuery

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    import ompl.base as ob


class NExtensions(ob.PlannerTerminationCondition):
    def __init__(self, max_n_extensions: int):
        super().__init__(ob.PlannerTerminationConditionFn(self.condition))
        self.max_n_extensions = max_n_extensions
        self.attempted_extensions = 0

    def condition(self):
        should_terminate = self.attempted_extensions >= self.max_n_extensions
        return should_terminate


def generate_pretransfer_examples(environment: Dict, state: Dict, batch_size: int, verbose: int):
    # this goal should be unreachable so the planner basically ignores it
    goal = {'point': np.array([1e9, 0, 0], dtype=np.float32)}
    planning_query = PlanningQuery(goal=goal,
                                   environment=environment,
                                   start=state,
                                   seed=0,
                                   trial_start_time_seconds=perf_counter())
    # run planner
    planner_params = {}
    planner = get_planner(planner_params=planner_params, verbose=verbose, log_full_tree=True)
    # create termination criterion based on batch size
    planner.ptc = NExtensions(max_n_extensions=batch_size)
    # we are not running the planner to reach a goal, so goal bias should be zero
    planner.rrt.setGoalBias(0)

    planning_result = planner.plan(planning_query=planning_query)

    # convert into examples for the classifier
    # planning_result.tree
    pass


def generate_initial_configs(initial_configs_dir: pathlib.Path):
    filenames = list(initial_configs_dir.glob("*.pkl"))
    for filename in itertools.cycle(filenames):
        with filename.open("rb") as file:
            initial_config = pickle.load(file)
        environment = initial_config['env']
        state = initial_config['state']
        yield environment, state


def generate_pretransfer_dataset(initial_configs_dir: pathlib.Path,
                                 n_examples: int,
                                 outdir: pathlib.Path,
                                 batch_size: int,
                                 verbose: int):
    assert n_examples % batch_size == 0
    files_dataset = FilesDataset(outdir)
    n_batches_of_examples = int(n_examples / batch_size)
    total_example_idx = 0
    t0 = perf_counter()
    last_t = perf_counter()

    initial_configs_generator = generate_initial_configs(initial_configs_dir)
    for batch_idx in range(n_batches_of_examples):

        # sample an initial configuration
        environment, state = next(initial_configs_generator)

        # generate examples
        for example in generate_pretransfer_examples(environment, state, batch_size, verbose):
            now = perf_counter()
            dt = now - last_t
            total_dt = now - t0
            last_t = now
            if verbose >= 0:
                print(f'Example {total_example_idx} dt={dt:.3f}, total time={total_dt:.3f}')

            tf_write_example(outdir, example, total_example_idx)
            total_example_idx += 1
            files_dataset.add(outdir)

    print("Splitting dataset")
    files_dataset.split()


@ros_init.with_ros("generate_pretransfer_dataset")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('initial_configs_dir', type=pathlib.Path)
    parser.add_argument('envs_dir', type=pathlib.Path)
    parser.add_argument('n_examples', type=pathlib.Path)
    parser.add_argument('outdir', type=pathlib.Path)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--verbose', '-v', action='count', default=0)

    args = parser.parse_args()

    generate_pretransfer_dataset(**vars(args))


if __name__ == '__main__':
    main()
