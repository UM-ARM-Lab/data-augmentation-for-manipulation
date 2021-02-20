from typing import Dict

from link_bot_classifiers import classifier_utils
from link_bot_planning.rrt import RRT
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.pycommon import paths_from_json
from state_space_dynamics import dynamics_utils, filter_utils


def get_planner(planner_params: Dict, verbose: int):
    # TODO: remove when backwards compatibility no longer needed
    if 'planner_type' not in planner_params:
        planner_type = 'rrt'
    else:
        planner_type = planner_params['planner_type']

    scenario = get_scenario(planner_params["scenario"])

    if planner_type == 'rrt':
        fwd_model = load_fwd_model(planner_params, scenario)
        filter_model = load_filter(planner_params, scenario)
        classifier_models = load_classifier(planner_params, scenario)

        action_params_with_defaults = fwd_model.data_collection_params
        action_params_with_defaults.update(planner_params['action_params'])
        planner = RRT(fwd_model=fwd_model,
                      filter_model=filter_model,
                      classifier_models=classifier_models,
                      planner_params=planner_params,
                      action_params=action_params_with_defaults,
                      scenario=scenario,
                      verbose=verbose)
    elif planner_type == 'shooting':
        fwd_model = load_fwd_model(planner_params)
        filter_model = filter_utils.load_filter(paths_from_json(planner_params['filter_model_dir']))

        from link_bot_planning.shooting_method import ShootingMethod

        action_params_with_defaults = fwd_model.data_collection_params
        action_params_with_defaults.update(planner_params['action_params'])
        planner = ShootingMethod(fwd_model=fwd_model,
                                 classifier_model=None,
                                 scenario=scenario,
                                 params={
                                     'n_samples': 1000
                                 },
                                 filter_model=filter_model,
                                 action_params=action_params_with_defaults)
    else:
        raise NotImplementedError(f"planner type {planner_type} not implemented")
    return planner


def load_classifier(planner_params: Dict, scenario: ExperimentScenario):
    classifier_model_dir = paths_from_json(planner_params['classifier_model_dir'])
    classifier_models = [classifier_utils.load_generic_model(d, scenario=scenario) for d in classifier_model_dir]
    return classifier_models


def load_filter(planner_params: Dict, scenario: ExperimentScenario):
    filter_model = filter_utils.load_filter(paths_from_json(planner_params['filter_model_dir']), scenario)
    return filter_model


def load_fwd_model(planner_params: Dict, scenario: ExperimentScenario):
    fwd_model_dirs = paths_from_json(planner_params['fwd_model_dir'])
    fwd_model, _ = dynamics_utils.load_generic_model(fwd_model_dirs, scenario)
    return fwd_model
