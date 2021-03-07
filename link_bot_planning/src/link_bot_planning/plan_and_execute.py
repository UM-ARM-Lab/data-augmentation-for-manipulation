#!/usr/bin/env python
import pathlib
import pickle
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
from colorama import Fore
from dataclasses_json import dataclass_json

import rospy
from arc_utilities.listener import Listener
from gazebo_msgs.msg import LinkStates
from jsk_recognition_msgs.msg import BoundingBox
from link_bot_classifiers import recovery_policy_utils
from link_bot_data import dataset_utils
from link_bot_data.dynamics_dataset import DynamicsDatasetLoader
from link_bot_planning.my_planner import MyPlannerStatus, PlanningQuery, PlanningResult, MyPlanner, SetupInfo
from link_bot_pycommon.base_services import BaseServices
from link_bot_pycommon.bbox_visualization import extent_to_bbox
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from moonshine.moonshine_utils import numpify, remove_batch, add_batch


class TrialStatus(Enum):
    Reached = "reached"
    Timeout = "timeout"
    NotProgressingNoRecovery = "not_progressing_no_recovery"


@dataclass_json
@dataclass
class ExecutionResult:
    path: List[Dict]
    end_trial: bool


@dataclass
class TrialResult:
    setup_info: SetupInfo
    planning_queries: List[PlanningQuery]
    total_time: float
    trial_status: TrialStatus
    trial_idx: int
    goal: Dict
    steps: int
    end_state: Dict


def execute_actions(
        scenario: ExperimentScenario,
        environment: Dict,
        start_state: Dict,
        actions: List[Dict],
        use_gt_rope: bool,
        plot: bool = False):
    pre_action_state = start_state
    actual_path = [pre_action_state]
    for action in actions:
        if plot:
            scenario.plot_environment_rviz(environment)
            scenario.plot_state_rviz(pre_action_state, label='actual')
            scenario.plot_executed_action(pre_action_state, action)

        end_trial = scenario.execute_action(action)
        state_t = scenario.get_state()
        if use_gt_rope:
            state_t = dataset_utils.use_gt_rope(state_t)

        actual_path.append(state_t)

        pre_action_state = state_t

    if plot:
        scenario.plot_environment_rviz(environment)
        scenario.plot_state_rviz(state_t, label='actual')

    return actual_path, end_trial


class PlanAndExecute:

    def __init__(self,
                 planner: MyPlanner,
                 trials: List[int],
                 verbose: int,
                 planner_params: Dict,
                 service_provider: BaseServices,
                 no_execution: bool,
                 use_gt_rope,
                 test_scenes_dir: Optional[pathlib.Path] = None,
                 seed: int = 0):
        self.use_gt_rope = use_gt_rope
        self.planner = planner
        self.scenario = self.planner.scenario
        self.trials = trials
        self.planner_params = planner_params
        self.verbose = verbose
        self.service_provider = service_provider
        self.no_execution = no_execution
        self.env_rng = np.random.RandomState(0)
        self.goal_rng = np.random.RandomState(0)
        self.recovery_rng = np.random.RandomState(0)
        self.seed = seed
        self.test_scenes_dir = test_scenes_dir
        if self.planner_params['recovery']['use_recovery']:
            recovery_model_dir = pathlib.Path(self.planner_params['recovery']['recovery_model_dir'])
            self.recovery_policy = recovery_policy_utils.load_generic_model(model_dir=recovery_model_dir,
                                                                            scenario=self.scenario,
                                                                            rng=self.recovery_rng,
                                                                            # FIXME: hacky is heck
                                                                            update_hparams={
                                                                                'extent': self.planner_params[
                                                                                              'extent']})
        else:
            self.recovery_policy = None

        self.n_failures = 0

        # for saving snapshots of the world
        self.link_states_listener = Listener("gazebo/link_states", LinkStates)

        # Debugging
        if self.verbose >= 2:
            self.goal_bbox_pub = rospy.Publisher('goal_bbox', BoundingBox, queue_size=10, latch=True)
            bbox_msg = extent_to_bbox(planner_params['goal_params']['extent'])
            bbox_msg.header.frame_id = 'world'
            self.goal_bbox_pub.publish(bbox_msg)

        goal_params = self.planner_params['goal_params']
        if goal_params['type'] == 'fixed':
            def _fixed_goal_gen(_: int, __: Dict):
                goal = numpify(goal_params['goal_fixed'])
                goal['goal_type'] = goal_params['goal_type']
                return goal

            self.goal_generator = _fixed_goal_gen
        elif goal_params['type'] == 'saved':
            def _saved_goals_gen(trial_idx: int, _: Dict):
                saved_goal_filename = self.test_scenes_dir / f'goal_{trial_idx:04d}.pkl'
                with saved_goal_filename.open('rb') as goal_file:
                    goal = pickle.load(goal_file)
                goal['goal_type'] = goal_params['goal_type']
                return goal

            self.goal_generator = _saved_goals_gen
        elif goal_params['type'] == 'random':
            def _rand_goal_gen(_: int, env: Dict):
                goal = self.scenario.sample_goal(environment=env,
                                                 rng=self.goal_rng,
                                                 planner_params=self.planner_params)
                goal['goal_type'] = goal_params['goal_type']
                return goal

            self.goal_generator = _rand_goal_gen
        elif goal_params['type'] == 'dataset':
            dataset = DynamicsDatasetLoader([pathlib.Path(goal_params['goals_dataset'])])
            tf_dataset = dataset.get_datasets(mode='val')
            goal_dataset_iterator = iter(tf_dataset)

            def _dataset_goal_gen(trial_idx: int, env: Dict):
                example = next(goal_dataset_iterator)
                example_t = dataset.index_time_batched(example_batched=add_batch(example), t=1)
                goal = remove_batch(example_t)
                goal['goal_type'] = goal_params['goal_type']
                return goal

            self.goal_generator = _dataset_goal_gen
        else:
            raise NotImplementedError(f"invalid goal param type {goal_params['type']}")

    def run(self):
        self.scenario.randomization_initialization(params=self.planner_params)
        for trial_idx in self.trials:
            self.plan_and_execute(trial_idx)

        self.on_complete()
        self.service_provider.play()

    def plan_and_execute(self, trial_idx: int):
        self.set_random_seeds_for_trial(trial_idx)

        setup_info = self.setup_test_scene(trial_idx)

        self.on_start_trial(trial_idx)

        start_time = time.perf_counter()
        total_timeout = self.planner_params['termination_criteria']['total_timeout']

        # Get the goal (default is to randomly sample one)
        environment = self.get_environment()
        goal = self.get_goal(trial_idx, environment)

        attempt_idx = 0
        steps_data = []
        planning_queries = []
        max_attempts = 20  # to save time, since it's unlikely we succeed after this many attempts
        while True:
            # get start states
            self.service_provider.play()
            start_state = self.scenario.get_state()
            self.service_provider.pause()
            if self.use_gt_rope:
                start_state = dataset_utils.use_gt_rope(start_state)

            # NOTE: we have assumed the environment does not change after executing, this is a performance optimization
            #  because getting the environment can be slow (~10 seconds)
            # environment = self.get_environment()

            # Try to make the seeds reproducible, but it needs to change based on attempt idx or we would just keep
            # trying the same plans over and over
            seed = 100000 * trial_idx + attempt_idx + self.seed
            planning_query = PlanningQuery(goal=goal,
                                           environment=environment,
                                           start=start_state,
                                           seed=seed,
                                           trial_start_time_seconds=start_time)
            planning_queries.append(planning_query)

            planning_result = self.plan(planning_query)

            time_since_start = time.perf_counter() - start_time

            if planning_result.status == MyPlannerStatus.Failure:
                raise RuntimeError("planning failed -- is the start state out of bounds?")
            elif planning_result.status == MyPlannerStatus.NotProgressing:
                if self.recovery_policy is None:
                    # Nothing else to do here, just give up
                    self.service_provider.play()
                    end_state = self.scenario.get_state()
                    self.service_provider.pause()
                    if self.use_gt_rope:
                        end_state = dataset_utils.use_gt_rope(end_state)
                    trial_status = TrialStatus.NotProgressingNoRecovery
                    trial_msg = f"Trial {trial_idx} Ended: not progressing, no recovery. {time_since_start:.3f}s"
                    rospy.loginfo(Fore.BLUE + trial_msg + Fore.RESET)
                    trial_data_dict = {
                        'setup_info':       setup_info,
                        'planning_queries': planning_queries,
                        'total_time':       time_since_start,
                        'trial_status':     trial_status,
                        'trial_idx':        trial_idx,
                        'goal':             goal,
                        'steps':            steps_data,
                        'end_state':        end_state,
                    }
                    self.on_trial_complete(trial_data_dict, trial_idx)
                    return
                else:
                    recovery_action = self.recovery_policy(environment=planning_query.environment,
                                                           state=planning_query.start)
                    attempt_idx += 1
                    if recovery_action is None:
                        rospy.loginfo(f"Could not sample a valid recovery action {attempt_idx}")
                        execution_result = ExecutionResult(path=[], end_trial=True)
                    else:
                        rospy.loginfo(f"Attempting recovery action {attempt_idx}")

                        if self.verbose >= 3:
                            rospy.loginfo("Chosen Recovery Action:")
                            rospy.loginfo(recovery_action)
                        self.service_provider.play()
                        execution_result = self.execute_recovery_action(recovery_action)
                        self.service_provider.pause()

                    # Extract planner data now before it goes out of scope (in C++)
                    steps_data.append({
                        'type':             'executed_recovery',
                        'planning_query':   planning_query,
                        'planning_result':  planning_result,
                        'recovery_action':  recovery_action,
                        'execution_result': execution_result,
                        'time_since_start': time_since_start,
                    })
            else:
                self.service_provider.play()
                execution_result = self.execute(planning_query, planning_result)
                self.service_provider.pause()
                steps_data.append({
                    'type':             'executed_plan',
                    'planning_query':   planning_query,
                    'planning_result':  planning_result,
                    'execution_result': execution_result,
                    'time_since_start': time_since_start,
                })
                self.on_execution_complete(planning_query, planning_result, execution_result)

            self.service_provider.play()
            end_state = self.scenario.get_state()
            self.service_provider.pause()
            if self.use_gt_rope:
                end_state = dataset_utils.use_gt_rope(end_state)

            d = self.scenario.distance_to_goal(end_state, planning_query.goal)
            rospy.loginfo(f"distance to goal after execution is {d:.3f}")
            reached_goal = (d <= self.planner_params['goal_params']['threshold'] + 1e-6)

            end_conditions = [
                reached_goal,
                time_since_start > total_timeout,
                self.no_execution,
                execution_result.end_trial,
                attempt_idx > max_attempts,
            ]
            end_trial = np.any(end_conditions)
            if end_trial:
                if reached_goal:
                    trial_status = TrialStatus.Reached
                    rospy.loginfo(Fore.BLUE + f"Trial {trial_idx} Ended: Goal reached!" + Fore.RESET)
                else:
                    trial_status = TrialStatus.Timeout
                    rospy.loginfo(Fore.BLUE + f"Trial {trial_idx} Ended: Timeout {time_since_start:.3f}s" + Fore.RESET)
                trial_data_dict = {
                    'setup_info':       setup_info,
                    'planning_queries': planning_queries,
                    'total_time':       time_since_start,
                    'trial_status':     trial_status,
                    'trial_idx':        trial_idx,
                    'goal':             goal,
                    'steps':            steps_data,
                    'end_state':        end_state,
                }
                self.on_trial_complete(trial_data_dict, trial_idx)
                return

    def setup_test_scene(self, trial_idx: int):
        if self.test_scenes_dir is not None:
            # Gazebo specific
            bagfile_name = self.test_scenes_dir / f'scene_{trial_idx:04d}.bag'
            rospy.loginfo(Fore.GREEN + f"Restoring scene {bagfile_name}")
            # rospy.logerr("skipping restore")
            self.scenario.restore_from_bag(self.service_provider, self.planner_params, bagfile_name)
            return SetupInfo(bagfile_name=bagfile_name)
        else:
            rospy.loginfo(Fore.GREEN + f"Randomizing Environment")
            self.randomize_environment()
            return SetupInfo(bagfile_name=None)

    def plan(self, planning_query: PlanningQuery):
        ############
        # Planning #
        ############
        if self.verbose >= 1:
            (Fore.MAGENTA + "Planning to {}".format(planning_query.goal) + Fore.RESET)
        planning_result = self.planner.plan(planning_query=planning_query)
        rospy.loginfo(f"Planning time: {planning_result.time:5.3f}s, Status: {planning_result.status}")

        self.on_plan_complete(planning_query, planning_result)

        return planning_result

    def execute(self, planning_query: PlanningQuery, planning_result: PlanningResult):
        # execute the plan, collecting the states that actually occurred
        self.on_before_execute()
        end_trial = False
        if self.no_execution:
            state_t = self.scenario.get_state()
            if self.use_gt_rope:
                state_t = dataset_utils.use_gt_rope(state_t)
            actual_path = [state_t]
        else:
            if self.verbose >= 2 and not self.no_execution:
                rospy.loginfo(Fore.CYAN + "Executing Plan" + Fore.RESET)
            actual_path, end_trial = execute_actions(scenario=self.scenario,
                                                     environment=planning_query.environment,
                                                     start_state=planning_query.start,
                                                     actions=planning_result.actions,
                                                     use_gt_rope=self.use_gt_rope,
                                                     plot=True)

        # post-execution callback
        execution_result = ExecutionResult(path=actual_path, end_trial=end_trial)
        return execution_result

    def execute_recovery_action(self, action: Dict):
        end_trial = False
        if self.no_execution:
            actual_path = []
        else:
            before_state = self.scenario.get_state()
            if self.use_gt_rope:
                before_state = dataset_utils.use_gt_rope(before_state)
            end_trial = self.scenario.execute_action(action)
            after_state = self.scenario.get_state()
            if self.use_gt_rope:
                after_state = dataset_utils.use_gt_rope(after_state)
            actual_path = [before_state, after_state]
        execution_result = ExecutionResult(path=actual_path, end_trial=end_trial)
        return execution_result

    def randomize_environment(self):
        self.scenario.randomize_environment(self.env_rng, self.planner_params)

    def get_environment(self):
        # get the environment, which here means anything which is assumed constant during planning
        get_env_params = self.planner_params.copy()
        get_env_params['res'] = self.planner.fwd_model.data_collection_params['res']
        return self.scenario.get_environment(get_env_params)

    def set_random_seeds_for_trial(self, trial_idx: int):
        self.env_rng.seed(trial_idx + self.seed)
        self.recovery_rng.seed(trial_idx + self.seed)
        self.goal_rng.seed(trial_idx + self.seed)

    def on_trial_complete(self, trial_data, trial_idx: int):
        pass

    def get_goal(self, trial_idx: int, environment: Dict):
        return self.goal_generator(trial_idx, environment)

    def on_plan_complete(self,
                         planning_query: PlanningQuery,
                         planning_result: PlanningResult):
        # visualize the plan
        if self.verbose >= 1:
            self.scenario.animate_final_path(environment=planning_query.environment,
                                             planned_path=planning_result.path,
                                             actions=planning_result.actions)

    def on_before_execute(self):
        pass

    def on_start_trial(self, trial_idx: int):
        pass

    def on_execution_complete(self,
                              planning_query: PlanningQuery,
                              planning_result: PlanningResult,
                              execution_result: ExecutionResult):
        pass

    def on_complete(self):
        pass
