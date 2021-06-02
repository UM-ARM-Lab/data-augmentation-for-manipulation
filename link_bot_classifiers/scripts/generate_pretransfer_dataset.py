import argparse
import itertools
import logging
import pathlib
import pickle
import warnings
from time import perf_counter
from typing import Dict

import tensorflow as tf

from arc_utilities import ros_init
from link_bot_data.dataset_utils import tf_write_example, add_predicted
from link_bot_data.files_dataset import FilesDataset
from link_bot_planning.floating_rope_ompl import NoGoal
from link_bot_planning.get_planner import get_planner
from link_bot_planning.my_planner import PlanningQuery, LoggingTree
from link_bot_planning.planning_evaluation import load_planner_params
from link_bot_planning.timeout_or_not_progressing import NExtensions
from link_bot_pycommon.marker_index_generator import marker_index_generator
from moonshine.moonshine_utils import sequence_of_dicts_to_dict_of_tensors, add_batch_single

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    import ompl.util as ou


class GeneratePretransferDataset:

    def __init__(self, initial_configs_dir: pathlib.Path,
                 planner_params_filename: pathlib.Path,
                 n_examples: int,
                 outdir: pathlib.Path,
                 batch_size: int,
                 verbose: int):
        self.initial_configs_dir = initial_configs_dir
        self.planner_params_filename = planner_params_filename
        self.verbose = verbose
        self.batch_size = batch_size
        self.n_examples = n_examples
        self.outdir = outdir

        self.planning_seed = 0
        self.total_example_idx = 0

        self.planner_params = load_planner_params(self.planner_params_filename)
        self.planner_params['smooth'] = False
        self.planner_params['classifier_model_dir'] = [pathlib.Path("cl_trials/new_feasibility_baseline/none")]

        self.planner = get_planner(planner_params=self.planner_params, verbose=self.verbose, log_full_tree=True)
        self.scenario = self.planner.scenario

        # create termination criterion based on batch size
        def set_ptc(*args, **kwargs):
            return NExtensions(max_n_extensions=self.batch_size)

        self.planner.make_ptc = set_ptc
        # we are not running the planner to reach a goal, so goal bias should be zero
        self.planner.rrt.setGoalBias(0)
        self.planner.goal_region = NoGoal(self.planner.si)

        def make_no_goal_region(*args, **kwargs):
            goal_region = NoGoal(self.planner.si)

            return goal_region

        self.planner.make_goal_region = make_no_goal_region

    def clear_markers(self):
        self.scenario.reset_planning_viz()

    def generate_pretransfer_examples(self, environment: Dict, state: Dict):
        self.clear_markers()

        # this goal should be unreachable so the planner basically ignores it
        goal = {}
        planning_query = PlanningQuery(goal=goal,
                                       environment=environment,
                                       start=state,
                                       seed=self.planning_seed,
                                       trial_start_time_seconds=perf_counter())

        planning_result = self.planner.plan(planning_query)
        self.planning_seed += 1

        # convert into examples for the classifier
        yield from self.dfs(environment, planning_result.tree)

    def dfs(self, environment: Dict, tree: LoggingTree, depth: int = 0):
        for child in tree.children:
            # if we only have one child we can skip the restore, this speeds things up a lot
            if self.verbose >= 1:
                self.visualize_example(action=child.action,
                                       before_state_predicted={add_predicted(k): v for k, v in
                                                               tree.state.items()},
                                       after_state_predicted={add_predicted(k): v for k, v in child.state.items()},
                                       environment=environment)

            yield from self.generate_example(
                environment=environment,
                action=child.action,
                before_state_pred=tree.state,
                after_state_pred=child.state,
                classifier_start_t=depth,
            )
            # recursion
            yield from self.dfs(environment, child, depth=depth + 1)

    def generate_example(self,
                         environment: Dict,
                         action: Dict,
                         before_state_pred: Dict,
                         after_state_pred: Dict,
                         classifier_start_t: int):
        classifier_horizon = 2  # this script only handles this case
        example_states_pred = sequence_of_dicts_to_dict_of_tensors([before_state_pred, after_state_pred])
        if 'num_diverged' in example_states_pred:
            example_states_pred.pop("num_diverged")
        example_actions = add_batch_single(action)
        example = {
            'classifier_start_t': classifier_start_t,
            'classifier_end_t':   classifier_start_t + classifier_horizon,
            'prediction_start_t': 0,
            'traj_idx':           self.total_example_idx,
            'time_idx':           [0, 1],
        }
        example.update(environment)
        example.update({add_predicted(k): v for k, v in example_states_pred.items()})
        example.update(example_actions)
        yield example

    def visualize_example(self,
                          action: Dict,
                          after_state_predicted: Dict,
                          before_state_predicted: Dict,
                          environment: Dict):
        self.scenario.plot_environment_rviz(environment)
        self.scenario.plot_state_rviz(before_state_predicted, idx=0, label='predicted',
                                      color='blue')
        self.scenario.plot_state_rviz(after_state_predicted, idx=1, label='predicted',
                                      color='blue')
        self.scenario.plot_action_rviz(before_state_predicted, action, idx=0, label='actual')

    def generate_initial_configs(self):
        filenames = list(self.initial_configs_dir.glob("*.pkl"))
        for filename in itertools.cycle(filenames):
            with filename.open("rb") as file:
                initial_config = pickle.load(file)
            environment = initial_config['env']
            state = initial_config['state']
            yield environment, state

    def generate(self):
        assert self.n_examples % self.batch_size == 0

        self.outdir.mkdir(exist_ok=True, parents=True)

        files_dataset = FilesDataset(self.outdir)
        n_batches_of_examples = int(self.n_examples / self.batch_size)
        t0 = perf_counter()
        last_t = perf_counter()

        initial_configs_generator = self.generate_initial_configs()
        for batch_idx in range(n_batches_of_examples):

            # sample an initial configuration
            environment, state = next(initial_configs_generator)

            # generate examples
            for example in self.generate_pretransfer_examples(environment, state):
                now = perf_counter()
                dt = now - last_t
                total_dt = now - t0
                last_t = now
                if self.verbose >= 0:
                    print(f'Example {self.total_example_idx} dt={dt:.3f}, total time={total_dt:.3f}')

                full_filename = tf_write_example(self.outdir, example, self.total_example_idx)
                self.total_example_idx += 1
                files_dataset.add(full_filename)

        print("Splitting dataset")
        files_dataset.split()


@ros_init.with_ros("generate_pretransfer_dataset")
def main():
    tf.get_logger().setLevel(logging.FATAL)
    ou.setLogLevel(ou.LOG_ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument('initial_configs_dir', type=pathlib.Path)
    parser.add_argument('planner_params_filename', type=pathlib.Path)
    parser.add_argument('n_examples', type=int)
    parser.add_argument('outdir', type=pathlib.Path)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--verbose', '-v', action='count', default=0)

    args = parser.parse_args()

    g = GeneratePretransferDataset(**vars(args))
    g.generate()


if __name__ == '__main__':
    main()
