#!/usr/bin/env python
import pathlib
from typing import Dict, List, Optional

import hjson
import rospkg
from colorama import Fore

import rospy
from link_bot_classifiers import train_test_classifier, train_test_recovery
from link_bot_data.base_collect_dynamics_data import TfDataCollector
from link_bot_data.classifier_dataset_utils import make_classifier_dataset_from_params_dict
from link_bot_data.recovery_dataset_utils import make_recovery_dataset_from_params_dict
from link_bot_planning.planning_evaluation import evaluate_multiple_planning
from link_bot_pycommon.get_service_provider import get_service_provider
from link_bot_pycommon.pycommon import paths_from_json
from link_bot_pycommon.serialization import MyHjsonEncoder
from state_space_dynamics import train_test

r = rospkg.RosPack()


class FullStackRunner:

    def __init__(self,
                 nickname: str,
                 unique_nickname: str,
                 full_stack_params: Dict,
                 launch: Optional[bool] = False,
                 gui: Optional[bool] = False,
                 verbose: Optional[int] = 0,
                 ):
        self.verbose = verbose
        self.launch = launch
        self.gui = gui
        self.full_stack_params = full_stack_params
        self.use_gt_rope = self.full_stack_params['use_gt_rope']
        self.nickname = nickname
        self.unique_nickname = unique_nickname
        service_provider_name = full_stack_params['service_provider']
        self.service_provider = get_service_provider(service_provider_name)

    def collect_dynamics_data_1(self, log: Dict, seed: int):
        collect_dynamics_1 = self.full_stack_params['collect_dynamics_1']
        scenario = collect_dynamics_1['scenario']
        collect_dynamics_data_params_filename = pathlib.Path(collect_dynamics_1['params'])
        link_bot_data_path = pathlib.Path(r.get_path('link_bot_data'))
        full_collect_dynamics_data_params_filename = link_bot_data_path / collect_dynamics_data_params_filename

        with full_collect_dynamics_data_params_filename.open('r') as collect_dynamics_data_params_file:
            collect_dynamics_data_params = hjson.load(collect_dynamics_data_params_file)

        if self.launch:
            self.service_provider.launch(collect_dynamics_1, gui=self.gui, world=collect_dynamics_1['world'])

        data_collector = TfDataCollector(scenario_name=scenario,
                                         service_provider=self.service_provider,
                                         params=collect_dynamics_data_params,
                                         seed=seed,
                                         verbose=self.verbose)
        dynamics_data_1_nickname = self.nickname + '_phase1'
        # this function will add a time stamp/git hash to the nickname
        files_dataset = data_collector.collect_data(robot_namespace=collect_dynamics_1['robot_namespace'],
                                                    n_trajs=collect_dynamics_1['n_trajs'],
                                                    nickname=dynamics_data_1_nickname)

        if self.launch:
            self.service_provider.kill()

        files_dataset.split()
        return {
            'dynamics_dataset_dir': files_dataset.root_dir,
        }

    def collect_dynamics_data_2(self, log: Dict, seed: int):
        collect_dynamics_2 = self.full_stack_params['collect_dynamics_2']
        scenario = collect_dynamics_2['scenario']
        collect_dynamics_data_params_filename = pathlib.Path(collect_dynamics_2['params'])
        link_bot_data_path = pathlib.Path(r.get_path('link_bot_data'))
        full_collect_dynamics_data_params_filename = link_bot_data_path / collect_dynamics_data_params_filename

        with full_collect_dynamics_data_params_filename.open('r') as collect_dynamics_data_params_file:
            collect_dynamics_data_params = hjson.load(collect_dynamics_data_params_file)

        if self.launch:
            self.service_provider.launch(collect_dynamics_2, gui=self.gui, world=collect_dynamics_2['world'])

        data_collector = TfDataCollector(scenario_name=scenario,
                                         service_provider=self.service_provider,
                                         params=collect_dynamics_data_params,
                                         seed=seed,
                                         verbose=self.verbose)
        dynamics_data_2_nickname = self.nickname + '_phase2'
        files_dataset = data_collector.collect_data(robot_namespace=collect_dynamics_2['robot_namespace'],
                                                    n_trajs=collect_dynamics_2['n_trajs'],
                                                    nickname=dynamics_data_2_nickname)

        if self.launch:
            self.service_provider.kill()

        files_dataset.split()
        return {
            'dynamics_dataset_dir': files_dataset.root_dir,
        }

    def learn_dynamics(self, log: Dict, seed: int):
        dynamics_dataset_dir = pathlib.Path(log['collect_dynamics_data_1']['dynamics_dataset_dir'])
        learn_dynamics_params = self.full_stack_params['learn_dynamics']
        n_ensemble = learn_dynamics_params['n_ensemble']
        batch_size = learn_dynamics_params['batch_size']
        epochs = learn_dynamics_params['epochs']
        state_space_dynamics_path = pathlib.Path(r.get_path('state_space_dynamics'))
        forward_model_hparams = state_space_dynamics_path / learn_dynamics_params['forward_model_hparams']

        trial_paths = []
        for ensemble_idx in range(n_ensemble):
            ensemble_seed = seed + ensemble_idx
            print(ensemble_seed)
            trial_path = train_test.train_main(dataset_dirs=[dynamics_dataset_dir],
                                               model_hparams=forward_model_hparams,
                                               trials_directory=pathlib.Path('dy_trials'),
                                               checkpoint=None,
                                               log=self.unique_nickname,
                                               batch_size=batch_size,
                                               epochs=epochs,
                                               seed=ensemble_seed,
                                               ensemble_idx=ensemble_idx,
                                               use_gt_rope=self.use_gt_rope,
                                               )
            trial_paths.append(trial_path)

        # Use one of the models we trained to compute the 90th percentile on the validation set
        classifier_threshold = train_test.compute_classifier_threshold(dataset_dirs=[dynamics_dataset_dir],
                                                                       checkpoint=trial_paths[0] / 'best_checkpoint',
                                                                       mode='val',
                                                                       use_gt_rope=self.use_gt_rope,
                                                                       batch_size=batch_size)
        return {
            'model_dirs':           trial_paths,
            'classifier_threshold': classifier_threshold
        }

    def learn_full_dynamics(self, log: Dict, seed: int):
        dynamics_dataset_2 = pathlib.Path(log['collect_dynamics_data_2']['dynamics_dataset_dir'])
        learn_dynamics_params = self.full_stack_params['learn_full_dynamics']
        n_ensemble = learn_dynamics_params['n_ensemble']
        batch_size = learn_dynamics_params['batch_size']
        epochs = learn_dynamics_params['epochs']
        state_space_dynamics_path = pathlib.Path(r.get_path('state_space_dynamics'))
        forward_model_hparams = state_space_dynamics_path / learn_dynamics_params['forward_model_hparams']

        # TODO: make use of use_gt_rope param

        trial_paths = []
        for ensemble_idx in range(n_ensemble):
            trial_path = train_test.train_main(dataset_dirs=[dynamics_dataset_2],
                                               model_hparams=forward_model_hparams,
                                               trials_directory=pathlib.Path('dy_trials'),
                                               checkpoint=None,
                                               log=self.unique_nickname + '_full',
                                               batch_size=batch_size,
                                               epochs=epochs,
                                               seed=seed,
                                               ensemble_idx=ensemble_idx,
                                               use_gt_rope=self.use_gt_rope,
                                               )
            trial_paths.append(trial_path)

        return {
            'model_dirs': trial_paths,
        }

    def make_classifier_dataset(self, log: Dict, seed: int):
        dynamics_dataset_2 = pathlib.Path(log['collect_dynamics_data_2']['dynamics_dataset_dir'])
        udnn_model_dirs = paths_from_json(log['learn_dynamics']['model_dirs'])
        make_classifier_dataset_params = self.full_stack_params['make_classifier_dataset']
        labeling_params_filename = pathlib.Path(make_classifier_dataset_params['labeling_params'])
        with labeling_params_filename.open("r") as labeling_params_file:
            labeling_params = hjson.load(labeling_params_file)
        labeling_params['threshold'] = log['learn_dynamics']['classifier_threshold']

        classifier_data_dir = pathlib.Path('classifier_data')
        classifier_data_dir.mkdir(exist_ok=True)
        outdir = classifier_data_dir / self.unique_nickname
        outdir.mkdir(exist_ok=True, parents=False)
        rospy.loginfo(Fore.GREEN + outdir.as_posix())
        classifier_dataset_dir = make_classifier_dataset_from_params_dict(dataset_dir=dynamics_dataset_2,
                                                                          fwd_model_dir=udnn_model_dirs,
                                                                          labeling_params=labeling_params,
                                                                          outdir=outdir,
                                                                          use_gt_rope=self.use_gt_rope,
                                                                          visualize=False,
                                                                          batch_size=8,
                                                                          )
        rospy.loginfo(Fore.GREEN + outdir.as_posix())
        return {
            'classifier_dataset_dir': classifier_dataset_dir,
        }

    def learn_classifier(self, log: Dict, seed: int):
        classifier_dataset_dir = pathlib.Path(log['make_classifier_dataset']['classifier_dataset_dir'])
        learn_classifier_params = self.full_stack_params['learn_classifier']
        batch_size = learn_classifier_params['batch_size']
        epochs = learn_classifier_params['epochs']
        classifiers_module_path = pathlib.Path(r.get_path('link_bot_classifiers'))
        classifier_hparams = classifiers_module_path / learn_classifier_params['classifier_hparams']
        trial_path, final_val_metrics = train_test_classifier.train_main(dataset_dirs=[classifier_dataset_dir],
                                                                         model_hparams=classifier_hparams,
                                                                         log=self.unique_nickname,
                                                                         trials_directory=pathlib.Path('cl_trials'),
                                                                         checkpoint=None,
                                                                         batch_size=batch_size,
                                                                         epochs=epochs,
                                                                         use_gt_rope=self.use_gt_rope,
                                                                         seed=seed)
        return {
            'model_dir': trial_path,
        }

    def make_recovery_dataset(self, log: Dict, seed: int):
        dynamics_dataset_2 = pathlib.Path(log['collect_dynamics_data_2']['dynamics_dataset_dir'])
        udnn_model_dirs = paths_from_json(log['learn_dynamics']['model_dirs'])
        classifier_model_dir = pathlib.Path(log['learn_classifier']['model_dir'])
        make_recovery_dataset_params = self.full_stack_params['make_recovery_dataset']
        labeling_params_filename = pathlib.Path(make_recovery_dataset_params['labeling_params'])
        with labeling_params_filename.open("r") as labeling_params_file:
            labeling_params = hjson.load(labeling_params_file)
        labeling_params['threshold'] = log['learn_dynamics']['classifier_threshold']
        recovery_data_dir = pathlib.Path('recovery_data')
        recovery_data_dir.mkdir(exist_ok=True)
        outdir = pathlib.Path('recovery_data') / self.unique_nickname
        outdir.mkdir(parents=False, exist_ok=True)
        batch_size = make_recovery_dataset_params['batch_size']
        recovery_dataset_dir = make_recovery_dataset_from_params_dict(dataset_dir=dynamics_dataset_2,
                                                                      fwd_model_dir=udnn_model_dirs,
                                                                      classifier_model_dir=classifier_model_dir,
                                                                      batch_size=batch_size,
                                                                      use_gt_rope=self.use_gt_rope,
                                                                      labeling_params=labeling_params,
                                                                      outdir=outdir)
        rospy.loginfo(Fore.GREEN + recovery_dataset_dir.as_posix())
        return {
            'recovery_dataset_dir': recovery_dataset_dir,
        }

    def learn_recovery(self, log: Dict, seed: int):
        recovery_dataset_dir = pathlib.Path(log['make_recovery_dataset']['recovery_dataset_dir'])
        classifier_model_dir = pathlib.Path(log['learn_classifier']['model_dir'])
        learn_recovery_params = self.full_stack_params['learn_recovery']
        batch_size = learn_recovery_params['batch_size']
        epochs = learn_recovery_params['epochs']
        recovery_module_path = pathlib.Path(r.get_path('link_bot_classifiers'))
        classifier_model_dir = classifier_model_dir / 'best_checkpoint'
        recovery_hparams = recovery_module_path / learn_recovery_params['recovery_hparams']
        trial_path = train_test_recovery.train_main(dataset_dirs=[recovery_dataset_dir],
                                                    classifier_checkpoint=classifier_model_dir,
                                                    model_hparams=recovery_hparams,
                                                    log=self.unique_nickname,
                                                    trials_directory=pathlib.Path('cl_trials'),
                                                    checkpoint=None,
                                                    batch_size=batch_size,
                                                    epochs=epochs,
                                                    seed=seed)
        rospy.loginfo(Fore.GREEN + trial_path.as_posix())
        return {
            'model_dir': trial_path,
        }

    def planning_evaluation(self, log: Dict, seed: int):
        logfile_name = log['filename']
        classifier_model_dir = pathlib.Path(log['learn_classifier']['model_dir'])

        udnn_model_dirs = paths_from_json(log['learn_dynamics']['model_dirs'])

        # full_dynamics_model_dirs = paths_from_json(log['learn_full_dynamics']['model_dirs'])
        full_dynamics_model_dirs = []

        recovery_model_dir = pathlib.Path(log['learn_recovery']['model_dir'])
        planning_module_path = pathlib.Path(r.get_path('link_bot_planning'))

        planning_evaluation_params = self.full_stack_params["planning_evaluation"]

        if "test_scenes_dir" in planning_evaluation_params:
            test_scenes_dir = pathlib.Path(planning_evaluation_params["test_scenes_dir"])
        else:
            test_scenes_dir = None

        n_trials = planning_evaluation_params['n_trials']
        trials = list(range(n_trials))

        planners_params_common_filename = pathlib.Path(planning_evaluation_params['planners_params_common'])
        planners_params = self.make_planners_params(classifier_model_dir,
                                                    full_dynamics_model_dirs,
                                                    udnn_model_dirs,
                                                    planners_params_common_filename,
                                                    planning_evaluation_params,
                                                    recovery_model_dir)

        if self.launch:
            self.service_provider.launch(planning_evaluation_params,
                                         gui=self.gui,
                                         world=planning_evaluation_params['world'])

        rospy.sleep(10)

        root = planning_module_path / 'results' / logfile_name.parent
        outdir = evaluate_multiple_planning(outdir=root,
                                            planners_params=planners_params,
                                            trials=trials,
                                            use_gt_rope=self.use_gt_rope,
                                            test_scenes_dir=test_scenes_dir,
                                            verbose=self.verbose,
                                            logfile_name=logfile_name,
                                            how_to_handle='retry',
                                            )

        if self.launch:
            self.service_provider.kill()

        rospy.loginfo(Fore.GREEN + outdir.as_posix())
        return outdir

    def make_planners_params(self,
                             classifier_model_dir: pathlib.Path,
                             full_dynamics_model_dirs: List[pathlib.Path],
                             udnn_model_dirs: List[pathlib.Path],
                             planners_params_common_filename: pathlib.Path,
                             planning_evaluation_params: Dict,
                             recovery_model_dir: pathlib.Path):
        planners_params = []
        with planners_params_common_filename.open('r') as planners_params_common_file:
            planner_params = hjson.load(planners_params_common_file)

        method_fwd_model_dirs = [d / 'best_checkpoint' for d in udnn_model_dirs]
        method_classifier_model_dir = [
            classifier_model_dir / 'best_checkpoint',
            'cl_trials/new_feasibility_baseline/none',  # the feasibility checker
        ]
        recovery = {
            'recovery_model_dir': recovery_model_dir / 'best_checkpoint',
            'use_recovery':       True,
        }

        method_name = self.nickname
        planner_params['fwd_model_dir'] = method_fwd_model_dirs
        planner_params['classifier_model_dir'] = method_classifier_model_dir
        planner_params['recovery'] = recovery
        planner_params['method_name'] = method_name

        planners_params.append((method_name, planner_params))
        return planners_params


def run_steps(fsr, full_stack_params, included_steps, log):
    logfile_name = log['filename']
    seed = full_stack_params['seed']
    if 'collect_dynamics_data_1' not in log and (included_steps is None or 'collect_dynamics_data_1' in included_steps):
        collect_dynamics_data_1_out = fsr.collect_dynamics_data_1(log, seed)
        log['collect_dynamics_data_1'] = collect_dynamics_data_1_out
        with logfile_name.open("w") as logfile:
            hjson.dump(log, logfile, cls=MyHjsonEncoder)
        rospy.loginfo(Fore.GREEN + logfile_name.as_posix())
    if 'collect_dynamics_data_2' not in log and (
            included_steps is None or 'collect_dynamics_data_2' in included_steps):
        collect_dynamics_data_2_out = fsr.collect_dynamics_data_2(log, seed)
        log['collect_dynamics_data_2'] = collect_dynamics_data_2_out
        with logfile_name.open("w") as logfile:
            hjson.dump(log, logfile, cls=MyHjsonEncoder)
        rospy.loginfo(Fore.GREEN + logfile_name.as_posix())
    if 'learn_dynamics' not in log and (included_steps is None or 'learn_dynamics' in included_steps):
        learn_dynamics_out = fsr.learn_dynamics(log, seed)
        log['learn_dynamics'] = learn_dynamics_out
        with logfile_name.open("w") as logfile:
            hjson.dump(log, logfile, cls=MyHjsonEncoder)
        rospy.loginfo(Fore.GREEN + logfile_name.as_posix())
    #
    # if 'learn_full_dynamics' not in log and (included_steps is None or 'learn_full_dynamics' in included_steps):
    #     learn_full_dynamics_out = fsr.learn_full_dynamics(log, seed)
    #     log['learn_full_dynamics'] = learn_full_dynamics_out
    #     with logfile_name.open("w") as logfile:
    #         hjson.dump(log, logfile, cls=MyHjsonEncoder)
    #     rospy.loginfo(Fore.GREEN + logfile_name.as_posix())
    if 'make_classifier_dataset' not in log and (included_steps is None or 'make_classifier_dataset' in included_steps):
        make_classifier_dataset_out = fsr.make_classifier_dataset(log, seed)
        log['make_classifier_dataset'] = make_classifier_dataset_out
        with logfile_name.open("w") as logfile:
            hjson.dump(log, logfile, cls=MyHjsonEncoder)
        rospy.loginfo(Fore.GREEN + logfile_name.as_posix())
    if 'learn_classifier' not in log and (included_steps is None or 'learn_classifier' in included_steps):
        learn_classifier_out = fsr.learn_classifier(log, seed)
        log['learn_classifier'] = learn_classifier_out
        with logfile_name.open("w") as logfile:
            hjson.dump(log, logfile, cls=MyHjsonEncoder)
        rospy.loginfo(Fore.GREEN + logfile_name.as_posix())
    if 'make_recovery_dataset' not in log and (included_steps is None or 'make_recovery_datastet' in included_steps):
        make_recovery_dataset_out = fsr.make_recovery_dataset(log, seed)
        log['make_recovery_dataset'] = make_recovery_dataset_out
        with logfile_name.open("w") as logfile:
            hjson.dump(log, logfile, cls=MyHjsonEncoder)
        rospy.loginfo(Fore.GREEN + logfile_name.as_posix())
    if 'learn_recovery' not in log and (included_steps is None or 'learn_recovery' in included_steps):
        learn_recovery_out = fsr.learn_recovery(log, seed)
        log['learn_recovery'] = learn_recovery_out
        with logfile_name.open("w") as logfile:
            hjson.dump(log, logfile, cls=MyHjsonEncoder)
        rospy.loginfo(Fore.GREEN + logfile_name.as_posix())
    if 'planning_evaluation' not in log and (included_steps is None or 'planning_evaluation' in included_steps):
        planning_evaluation_out = fsr.evaluate_multiple_planning(log, seed)
        log['planning_evaluation'] = planning_evaluation_out
        with logfile_name.open("w") as logfile:
            hjson.dump(log, logfile, cls=MyHjsonEncoder)
        rospy.loginfo(Fore.GREEN + logfile_name.as_posix())
