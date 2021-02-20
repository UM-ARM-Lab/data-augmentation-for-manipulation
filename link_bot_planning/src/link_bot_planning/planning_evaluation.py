import pathlib
import tempfile
import uuid
from time import time, sleep
from typing import Optional, Dict, List, Tuple

import numpy as np
from colorama import Fore
from ompl import util as ou

import rosbag
import rospy
from arc_utilities.algorithms import is_list_unique
from link_bot_gazebo_python import gazebo_services
from link_bot_planning import plan_and_execute
from link_bot_planning.get_planner import get_planner
from link_bot_planning.my_planner import MyPlanner
from link_bot_pycommon.base_services import BaseServices
from link_bot_pycommon.job_chunking import JobChunker
from link_bot_pycommon.pycommon import deal_with_exceptions
from link_bot_pycommon.serialization import dump_gzipped_pickle, my_dump
from moonshine.moonshine_utils import numpify


class EvalPlannerConfigs(plan_and_execute.PlanAndExecute):

    def __init__(self,
                 planner: MyPlanner,
                 service_provider: BaseServices,
                 job_chunker: JobChunker,
                 trials: List[int],
                 verbose: int,
                 planner_params: Dict,
                 outdir: pathlib.Path,
                 use_gt_rope,
                 record: Optional[bool] = False,
                 no_execution: Optional[bool] = False,
                 test_scenes_dir: Optional[pathlib.Path] = None,
                 saved_goals_filename: Optional[pathlib.Path] = None,
                 seed: int = 0,
                 ):
        super().__init__(planner, trials=trials, verbose=verbose, planner_params=planner_params,
                         service_provider=service_provider, no_execution=no_execution, use_gt_rope=use_gt_rope,
                         test_scenes_dir=test_scenes_dir, seed=seed)
        self.record = record
        self.outdir = outdir
        self.job_chunker = job_chunker

        self.outdir.mkdir(parents=True, exist_ok=True)
        rospy.loginfo(Fore.BLUE + f"Output directory: {self.outdir.as_posix()}")

        metadata = {
            "trials":         self.trials,
            "planner_params": self.planner_params,
            "scenario":       self.planner.scenario.simple_name(),
        }
        metadata.update(self.planner.get_metadata())
        with (self.outdir / 'metadata.json').open("w") as metadata_file:
            my_dump(metadata, metadata_file, indent=2)

        self.bag = None
        self.final_execution_to_goal_errors = []

    def randomize_environment(self):
        if self.verbose >= 1:
            rospy.loginfo("Randomizing env")
        self.service_provider.play()
        super().randomize_environment()
        self.service_provider.pause()
        if self.verbose >= 1:
            rospy.loginfo("End randomizing env")

    def on_start_trial(self, trial_idx: int):
        if self.record:
            filename = self.outdir.absolute() / 'plan-{}.avi'.format(trial_idx)
            self.service_provider.start_record_trial(str(filename))
            bagname = self.outdir.absolute() / f"follow_joint_trajectory_goal_{trial_idx}.bag"
            rospy.loginfo(Fore.YELLOW + f"Saving bag file name: {bagname.as_posix()}")
            self.bag = rosbag.Bag(bagname, 'w')

    def follow_joint_trajectory_goal_callback(self, goal_msg):
        if self.record:
            self.bag.write('/both_arms_controller/follow_joint_trajectory/goal', goal_msg)
            self.bag.flush()

    def on_trial_complete(self, trial_data: Dict, trial_idx: int):
        extra_trial_data = {
            "planner_params": self.planner_params,
            "scenario":       self.planner_params['scenario'],
            'current_time':   int(time()),
            'uuid':           uuid.uuid4(),
        }
        trial_data.update(extra_trial_data)
        data_filename = self.outdir / f'{trial_idx}_metrics.pkl.gz'
        dump_gzipped_pickle(trial_data, data_filename)

        if self.record:
            # TODO: maybe make this happen async?
            sleep(1)
            self.service_provider.stop_record_trial()
            self.bag.close()

        # compute the current running average success
        goal = trial_data['planning_queries'][0].goal
        final_actual_state = numpify(trial_data['end_state'])
        final_execution_to_goal_error = self.planner.scenario.distance_to_goal(final_actual_state, goal)
        self.final_execution_to_goal_errors.append(final_execution_to_goal_error)
        goal_threshold = self.planner_params['goal_params']['threshold']
        n = len(self.final_execution_to_goal_errors)
        success_percentage = np.count_nonzero(np.array(self.final_execution_to_goal_errors) < goal_threshold) / n * 100
        update_msg = f"[{self.outdir.name}] Current average success rate {success_percentage:.2f}%"
        rospy.loginfo(Fore.LIGHTBLUE_EX + update_msg)

        jobkey = self.jobkey(trial_idx)
        self.job_chunker.store_result(jobkey, {'data_filename': data_filename})

    @staticmethod
    def jobkey(trial_idx):
        jobkey = f'{trial_idx}'
        return jobkey

    def plan_and_execute(self, trial_idx: int):
        jobkey = self.jobkey(trial_idx)
        if self.job_chunker.result_exists(jobkey):
            rospy.loginfo(f"Found existing trial {jobkey}, skipping.")
            return
        super().plan_and_execute(trial_idx=trial_idx)


def evaluate_planning_method(planner_params: Dict,
                             job_chunker: JobChunker,
                             trials: List[int],
                             comparison_root_dir: pathlib.Path,
                             use_gt_rope: bool,
                             verbose: int = 0,
                             record: bool = False,
                             no_execution: bool = False,
                             timeout: Optional[int] = None,
                             test_scenes_dir: Optional[pathlib.Path] = None,
                             saved_goals_filename: Optional[pathlib.Path] = None,
                             seed: int = 0,
                             ):
    # override some arguments
    if timeout is not None:
        rospy.loginfo(f"Overriding with timeout {timeout}")
        planner_params["termination_criteria"]['timeout'] = timeout
        planner_params['total_timeout'] = timeout

    # Start Services
    service_provider = gazebo_services.GazeboServices()
    service_provider.play()  # time needs to be advancing while we setup the planner so it can use ROS to query things
    planner = get_planner(planner_params=planner_params, verbose=verbose)

    service_provider.setup_env(verbose=verbose,
                               real_time_rate=planner_params['real_time_rate'],
                               max_step_size=planner.fwd_model.max_step_size,
                               play=True)

    # FIXME: RAII -- you should not be able to call get_state on a scenario until this method has been called
    #  which could be done by making a type, something like "EmbodiedScenario" which has get_state and execute_action,
    planner.scenario.on_before_get_state_or_execute_action()

    runner = EvalPlannerConfigs(planner=planner,
                                service_provider=service_provider,
                                job_chunker=job_chunker,
                                trials=trials,
                                verbose=verbose,
                                planner_params=planner_params,
                                outdir=comparison_root_dir,
                                use_gt_rope=use_gt_rope,
                                record=record,
                                no_execution=no_execution,
                                test_scenes_dir=test_scenes_dir,
                                saved_goals_filename=saved_goals_filename,
                                seed=seed)
    return runner.run()


def planning_evaluation(outdir: pathlib.Path,
                        planners_params: List[Tuple[str, Dict]],
                        trials: List[int],
                        logfile_name: Optional[str],
                        use_gt_rope: bool,
                        start_idx: int = 0,
                        stop_idx: int = -1,
                        on_exception: Optional[str] = 'raise',
                        verbose: int = 0,
                        record: bool = False,
                        no_execution: bool = False,
                        timeout: Optional[int] = None,
                        test_scenes_dir: Optional[pathlib.Path] = None,
                        saved_goals_filename: Optional[pathlib.Path] = None,
                        seed: int = 0,
                        ):
    ou.setLogLevel(ou.LOG_ERROR)

    if logfile_name is None:
        logfile_name = pathlib.Path(tempfile.gettempdir()) / f'planning-evaluation-log-file-{time()}'

    job_chunker = JobChunker(logfile_name=logfile_name)

    rospy.loginfo(Fore.CYAN + "common output directory: {}".format(outdir))
    if not outdir.is_dir():
        rospy.loginfo(Fore.YELLOW + "Creating output directory: {}".format(outdir))
        outdir.mkdir(parents=True)

    # NOTE: if method names are not unique, we would overwrite results. Very bad!
    assert_method_names_are_unique(planners_params)

    for comparison_idx, (method_name, planner_params) in enumerate(planners_params):
        if comparison_idx < start_idx:
            continue
        if stop_idx != -1 and comparison_idx >= stop_idx:
            break

        job_chunker.setup_key(method_name)
        sub_job_chunker = job_chunker.sub_chunker(method_name)

        rospy.loginfo(Fore.GREEN + f"Running method {method_name}")
        comparison_root_dir = outdir / method_name

        deal_with_exceptions(on_exception=on_exception,
                             function=evaluate_planning_method,
                             planner_params=planner_params,
                             job_chunker=sub_job_chunker,
                             use_gt_rope=use_gt_rope,
                             trials=trials,
                             comparison_root_dir=comparison_root_dir,
                             verbose=verbose,
                             record=record,
                             no_execution=no_execution,
                             timeout=timeout,
                             test_scenes_dir=test_scenes_dir,
                             saved_goals_filename=saved_goals_filename,
                             print_exception=True,
                             seed=seed,
                             )

        rospy.loginfo(f"Results written to {outdir}")

    return outdir


def assert_method_names_are_unique(planners_params):
    method_names = [method_name for method_name, _ in planners_params]
    method_names_are_unique = is_list_unique(method_names)
    assert method_names_are_unique, f"Method names must be unique, but are {method_names}"
