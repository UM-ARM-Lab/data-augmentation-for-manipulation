import pathlib
import traceback
import uuid
from time import time, sleep
from typing import Optional, Dict, List, Tuple

import rosbag
import rospy
from colorama import Fore
from ompl import util as ou

from control_msgs.msg import FollowJointTrajectoryActionGoal
from link_bot_data.link_bot_dataset_utils import data_directory
from link_bot_gazebo_python import gazebo_services
from link_bot_planning import plan_and_execute
from link_bot_planning.get_planner import get_planner
from link_bot_planning.my_planner import MyPlanner
from link_bot_pycommon.base_services import BaseServices
from link_bot_pycommon.serialization import dummy_proof_write, my_dump


class EvalPlannerConfigs(plan_and_execute.PlanAndExecute):

    def __init__(self,
                 planner: MyPlanner,
                 service_provider: BaseServices,
                 planner_config_name: str,
                 trials: List[int],
                 verbose: int,
                 planner_params: Dict,
                 comparison_item_idx: int,
                 goal,
                 outdir: pathlib.Path,
                 record: Optional[bool] = False,
                 no_execution: Optional[bool] = False,
                 ):
        super().__init__(planner,
                         trials=trials,
                         verbose=verbose,
                         planner_params=planner_params,
                         service_provider=service_provider,
                         no_execution=no_execution)
        self.record = record
        self.planner_config_name = planner_config_name
        self.outdir = outdir

        self.subfolder = "{}_{}".format(
            self.planner_config_name, comparison_item_idx)
        self.root = self.outdir / self.subfolder
        self.root.mkdir(parents=True)
        print(Fore.CYAN + str(self.root) + Fore.RESET)
        self.goal = goal

        metadata = {
            "trials": self.trials,
            "planner_params": self.planner_params,
            "scenario": self.planner.scenario.simple_name(),
            "horizon": self.planner.classifier_model.horizon,
        }
        with (self.root / 'metadata.json').open("w") as metadata_file:
            my_dump(metadata, metadata_file, indent=2)

        self.joint_goal_sub = rospy.Subscriber("/both_arms_controller/follow_joint_trajectory/goal",
                                               FollowJointTrajectoryActionGoal,
                                               self.follow_joint_trajectory_goal_callback,
                                               queue_size=10)
        self.bag = None

    def randomize_environment(self):
        super().randomize_environment()

    def on_start_trial(self, trial_idx: int):
        if self.record:
            filename = self.root.absolute() / 'plan-{}.avi'.format(trial_idx)
            self.service_provider.start_record_trial(str(filename))
            bagname = self.root.absolute() / f"follow_joint_trajectory_goal_{trial_idx}.bag"
            print(Fore.YELLOW + str(bagname) + Fore.RESET)
            self.bag = rosbag.Bag(bagname, 'w')

    def follow_joint_trajectory_goal_callback(self, goal_msg):
        if self.record:
            self.bag.write('/both_arms_controller/follow_joint_trajectory/goal', goal_msg)
            self.bag.flush()

    def get_goal(self, environment: Dict):
        if self.goal is not None:
            if self.verbose >= 1:
                print("Using Goal {}".format(self.goal))
            return self.goal
        else:
            return super().get_goal(environment)

    def on_trial_complete(self, trial_data: Dict, trial_idx: int):
        extra_trial_data = {
            "planner_params": self.planner_params,
            "scenario": self.planner.scenario.simple_name(),
            'current_time': int(time()),
            'uuid': uuid.uuid4(),
        }
        trial_data.update(extra_trial_data)
        data_filename = self.root / f'{trial_idx}_metrics.json.gz'
        dummy_proof_write(trial_data, data_filename)

        if self.record:
            # TODO: maybe make this happen async?
            sleep(1)
            self.service_provider.stop_record_trial()
            self.bag.close()


def evaluate_planning_method(comparison_idx: int,
                             planner_params: Dict,
                             trials: List[int],
                             planner_config_name: str,
                             common_output_directory: pathlib.Path,
                             verbose: int = 0,
                             record: bool = False,
                             no_execution: bool = False,
                             timeout: Optional[int] = None,
                             ):
    # override some arguments
    if timeout is not None:
        rospy.loginfo(f"Overriding with timeout {timeout}")
        planner_params["termination_criteria"]['timeout'] = timeout

    # Start Services
    service_provider = gazebo_services.GazeboServices()
    planner, _ = get_planner(planner_params=planner_params,
                             verbose=verbose)

    service_provider.setup_env(verbose=verbose,
                               real_time_rate=planner_params['real_time_rate'],
                               max_step_size=planner.fwd_model.max_step_size)

    print(Fore.GREEN + f"Running Trials {trials}" + Fore.RESET)

    runner = EvalPlannerConfigs(
        planner=planner,
        service_provider=service_provider,
        planner_config_name=planner_config_name,
        trials=trials,
        verbose=verbose,
        planner_params=planner_params,
        outdir=common_output_directory,
        comparison_item_idx=comparison_idx,
        goal=planner_params['fixed_goal'],
        record=record,
        no_execution=no_execution
    )
    runner.run()


def planning_evaluation(root: pathlib.Path,
                        planners_params: List[Tuple[str, Dict]],
                        trials: List[int],
                        skip_on_exception: Optional[bool] = False,
                        verbose: int = 0,
                        record: bool = False,
                        no_execution: bool = False,
                        timeout: Optional[int] = None,
                        ):
    ou.setLogLevel(ou.LOG_ERROR)

    common_output_directory = data_directory(root)
    common_output_directory = pathlib.Path(common_output_directory)
    print(Fore.CYAN + "common output directory: {}".format(common_output_directory) + Fore.RESET)
    if not common_output_directory.is_dir():
        print(Fore.YELLOW + "Creating output directory: {}".format(common_output_directory) + Fore.RESET)
        common_output_directory.mkdir(parents=True)

    for comparison_idx, (planner_config_name, planner_params) in enumerate(planners_params):
        if skip_on_exception:
            try:
                evaluate_planning_method(comparison_idx=comparison_idx,
                                         planner_params=planner_params,
                                         trials=trials,
                                         planner_config_name=planner_config_name,
                                         common_output_directory=common_output_directory,
                                         verbose=verbose,
                                         record=record,
                                         no_execution=no_execution,
                                         timeout=timeout,
                                         )
            except Exception as e:
                traceback.print_exc()
                print()
        else:
            evaluate_planning_method(comparison_idx=comparison_idx,
                                     planner_params=planner_params,
                                     trials=trials,
                                     planner_config_name=planner_config_name, common_output_directory=common_output_directory)
        print(f"Results written to {common_output_directory}")

    return root
