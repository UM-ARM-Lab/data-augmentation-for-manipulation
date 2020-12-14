import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
from matplotlib import cm

from link_bot_planning.my_planner import MyPlannerStatus, PlanningQuery, PlanningResult, MyPlanner
from link_bot_planning.trajectory_optimizer import TrajectoryOptimizer
from link_bot_pycommon.dual_arm_sim_rope_scenario import SimDualArmRopeScenario
from link_bot_pycommon.floating_rope_ompl import FloatingRopeOmpl
from link_bot_pycommon.rope_dragging_ompl import RopeDraggingOmpl
from link_bot_pycommon.rope_dragging_scenario import RopeDraggingScenario
from link_bot_pycommon.scenario_ompl import ScenarioOmpl
from state_space_dynamics.base_filter_function import BaseFilterFunction

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    import ompl.base as ob
    import ompl.control as oc

import rospy
from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
# from link_bot_classifiers.collision_checker_classifier import CollisionCheckerClassifier
from link_bot_planning.ompl_viz import planner_data_to_json
from link_bot_planning.timeout_or_not_progressing import TimeoutOrNotProgressing
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from moonshine.tests.testing_utils import are_dicts_close_np
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


def get_ompl_scenario(scenario) -> ScenarioOmpl:
    if isinstance(scenario, RopeDraggingScenario):
        return RopeDraggingOmpl(scenario)
    elif isinstance(scenario, SimDualArmRopeScenario):
        return FloatingRopeOmpl(scenario)
    else:
        raise NotImplementedError(f"unimplemented scenario {scenario}")


class RRT(MyPlanner):

    def __init__(self,
                 fwd_model: BaseDynamicsFunction,
                 filter_model: BaseFilterFunction,
                 classifier_model: BaseConstraintChecker,
                 planner_params: Dict,
                 action_params: Dict,
                 scenario: ExperimentScenario,
                 verbose: int,
                 ):
        super().__init__(scenario=scenario, fwd_model=fwd_model, filter_model=filter_model)
        self.verbose = verbose
        self.fwd_model = fwd_model
        self.classifier_model = classifier_model
        self.params = planner_params
        self.action_params = action_params
        # TODO: consider making full env params h/w come from elsewhere. res should match the model though.
        self.classifier_rng = np.random.RandomState(0)
        self.state_sampler_rng = np.random.RandomState(0)
        self.goal_sampler_rng = np.random.RandomState(0)
        self.control_sampler_rng = np.random.RandomState(0)
        self.scenario = scenario
        self.scenario_ompl = get_ompl_scenario(self.scenario)

        self.state_space = self.scenario_ompl.make_ompl_state_space(planner_params=self.params,
                                                                    state_sampler_rng=self.state_sampler_rng,
                                                                    plot=self.verbose >= 2)
        self.control_space = self.scenario_ompl.make_ompl_control_space(self.state_space,
                                                                        self.control_sampler_rng,
                                                                        action_params=self.action_params)

        self.ss = oc.SimpleSetup(self.control_space)

        self.si: oc.SpaceInformation = self.ss.getSpaceInformation()

        def _dcs_allocator(si):
            # noinspection PyUnusedLocal
            def _cost_function(actions: List[Dict], environment: Dict, goal_state: Dict, mean_predictions: List[Dict]):
                del actions, environment
                return self.scenario.distance_to_goal_state(state=mean_predictions[1],
                                                            goal_type=self.params['goal_params']['goal_type'],
                                                            goal_state=goal_state)

            opt = TrajectoryOptimizer(fwd_model=self.fwd_model,
                                      classifier_model=None,
                                      scenario=self.scenario,
                                      params=self.params,
                                      verbose=self.verbose,
                                      cost_function=_cost_function)
            return self.scenario_ompl.make_directed_control_sampler(si,
                                                                    rng=self.control_sampler_rng,
                                                                    action_params=action_params,
                                                                    opt=opt)

        self.si.setDirectedControlSamplerAllocator(oc.DirectedControlSamplerAllocator(_dcs_allocator))

        self.ss.setStatePropagator(oc.AdvancedStatePropagatorFn(self.propagate))
        self.ss.setMotionsValidityChecker(oc.MotionsValidityCheckerFn(self.motions_valid))
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_valid))

        self.cleanup_before_plan(0)

        # just for debugging
        # self.cc = CollisionCheckerClassifier([pathlib.Path("cl_trials/cc_baseline/cc")], self.scenario, 0.0)
        # self.cc_but_accept_count = 0

        self.rrt = oc.RRT(self.si)
        self.rrt.setIntermediateStates(True)  # this is necessary, because we use this to generate datasets
        self.ss.setPlanner(self.rrt)
        self.si.setMinMaxControlDuration(1, 50)

    def cleanup_before_plan(self, seed):
        self.ptc = None
        self.n_total_action = None
        self.goal_region = None
        # a Dictionary containing the parts of state which are not predicted/planned for, i.e. the environment
        self.environment = None
        self.start_state = None
        self.closest_state_to_goal = None
        self.min_dist_to_goal = 10000

        self.classifier_rng.seed(seed)
        self.state_sampler_rng.seed(seed)
        self.goal_sampler_rng.seed(seed)
        self.control_sampler_rng.seed(seed)

        # just for debugging
        self.cc_but_accept_count = 0

    def is_valid(self, state):
        valid = self.state_space.satisfiesBounds(state)
        return valid

    def motions_valid(self, motions):
        print(".", end='', flush=True)
        final_state = self.scenario_ompl.ompl_state_to_numpy(motions[-1].getState())

        motions_valid = final_state['num_diverged'] < self.classifier_model.horizon - 1  # yes, minus 1
        motions_valid = bool(np.squeeze(motions_valid))
        if not motions_valid:
            if self.verbose >= 2:
                self.scenario.plot_rejected_state(final_state)

        # PTC bookkeeping to figure out how the planner is progressing
        self.ptc.attempted_extensions += 1
        if motions_valid:
            self.ptc.all_rejected = False
            dist_to_goal = self.scenario.distance_to_goal(final_state, self.goal_region.goal)
            if dist_to_goal < self.min_dist_to_goal:
                self.min_dist_to_goal = dist_to_goal
                self.closest_state_to_goal = final_state
                self.scenario.plot_state_closest_to_goal(final_state)
        # end PTC bookkeeping
        return motions_valid

    def motions_to_numpy(self, motions):
        states_sequence = []
        actions = []
        for t, motion in enumerate(motions):
            # motions is a vector of oc.Motion, which has a state, parent, and control
            state = motion.getState()
            control = motion.getControl()
            state_t = self.scenario_ompl.ompl_state_to_numpy(state)
            states_sequence.append(state_t)
            if t > 0:  # skip the first (null) action, because that would represent the action that brings us to the first state
                actions.append(self.scenario_ompl.ompl_control_to_numpy(state, control))
        actions = np.array(actions)
        return states_sequence, actions

    def predict(self, previous_states, previous_actions, new_action):
        new_actions = [new_action]
        last_previous_state = previous_states[-1]
        # t0 = perf_counter()
        mean_predicted_states, stdev_predicted_states = self.fwd_model.propagate(environment=self.environment,
                                                                                 start_states=last_previous_state,
                                                                                 actions=new_actions)
        # print('fwd model propagate', perf_counter() - t0) # takes 3ms
        # get only the final state predicted
        final_predicted_state = mean_predicted_states[-1]

        # compute new num_diverged by checking the constraint
        # walk back up the branch until num_diverged == 0
        all_states = [final_predicted_state]
        all_actions = [new_action]
        for previous_idx in range(len(previous_states) - 1, -1, -1):
            previous_state = previous_states[previous_idx]
            all_states.insert(0, previous_state)
            if previous_state['num_diverged'] == 0:
                break
            # this goes after the break because action_i brings you TO state_i and we don't want that last action
            previous_action = previous_actions[previous_idx - 1]
            all_actions.insert(0, previous_action)
        classifier_probabilities, _ = self.classifier_model.check_constraint(environment=self.environment,
                                                                             states_sequence=all_states,
                                                                             actions=all_actions)
        # not_in_collision = self.cc.check_constraint(environment=self.environment,
        #                                             states_sequence=all_states,
        #                                             actions=all_actions)
        final_classifier_probability = classifier_probabilities[-1]
        # if not not_in_collision and final_classifier_probability > 0.5:
        #     self.cc_but_accept_count += 1
        #     if self.verbose >= 2:
        #         self.scenario.plot_state_rviz(final_predicted_state,
        #                                       color='y',
        #                                       label='accepted in collision',
        #                                       idx=self.cc_but_accept_count)
        #
        if self.verbose >= 2:
            self.scenario.plot_accept_probability(final_classifier_probability)
        if final_classifier_probability > self.params['accept_threshold']:
            final_predicted_state['num_diverged'] = np.array([0.0])
        else:
            final_predicted_state['num_diverged'] = last_previous_state['num_diverged'] + 1
        return final_predicted_state, final_classifier_probability

    def propagate(self, motions, control, duration, state_out):
        del duration  # unused, multi-step propagation is handled inside propagateMotionsWhileValid

        # Convert from OMPL -> Numpy
        previous_states, previous_actions = self.motions_to_numpy(motions)
        previous_state = previous_states[-1]
        previous_ompl_state = motions[-1].getState()
        new_action = self.scenario_ompl.ompl_control_to_numpy(previous_ompl_state, control)
        np_final_state, final_classifier_probability = self.predict(previous_states, previous_actions, new_action)

        # Convert back Numpy -> OMPL
        self.scenario_ompl.numpy_to_ompl_state(np_final_state, state_out)

        if self.verbose >= 2:
            alpha = final_classifier_probability * 0.8 + 0.2
            classifier_probability_color = cm.Reds_r(final_classifier_probability)
            if len(previous_actions) == 0:
                random_color = cm.Dark2(self.control_sampler_rng.uniform(0, 1))
                RRT.propagate.r = random_color[0]
                RRT.propagate.g = random_color[1]
                RRT.propagate.b = random_color[2]
            elif not are_dicts_close_np(previous_actions[-1], new_action):
                random_color = cm.Dark2(self.control_sampler_rng.uniform(0, 1))
                RRT.propagate.r = random_color[0]
                RRT.propagate.g = random_color[1]
                RRT.propagate.b = random_color[2]

            statisfies_bounds = self.state_space.satisfiesBounds(state_out)
            if final_classifier_probability > 0.5 and statisfies_bounds:
                self.scenario.plot_tree_state(np_final_state, color=classifier_probability_color)
            self.scenario.plot_current_tree_state(
                np_final_state, horizon=self.classifier_model.horizon, color=classifier_probability_color)

            self.scenario.plot_tree_action(previous_state,
                                           new_action,
                                           r=RRT.propagate.r,
                                           g=RRT.propagate.g,
                                           b=RRT.propagate.b,
                                           a=alpha)

    propagate.r = 0
    propagate.g = 0
    propagate.b = 0

    def plan(self, planning_query: PlanningQuery):
        self.cleanup_before_plan(planning_query.seed)

        self.environment = planning_query.environment

        self.goal_region = self.scenario_ompl.make_goal_region(self.si,
                                                               rng=self.goal_sampler_rng,
                                                               params=self.params,
                                                               goal=planning_query.goal,
                                                               plot=self.verbose >= 2)

        # create start and goal states
        start_state = planning_query.start
        start_state['stdev'] = np.array([0.0])
        start_state['num_diverged'] = np.array([0.0])
        self.start_state = start_state
        ompl_start_scoped = ob.State(self.state_space)
        self.scenario_ompl.numpy_to_ompl_state(start_state, ompl_start_scoped())

        # visualization
        self.scenario.reset_planning_viz()
        self.scenario.plot_environment_rviz(planning_query.environment)
        self.scenario.plot_start_state(start_state)
        self.scenario.plot_goal_rviz(planning_query.goal, self.params['goal_params']['threshold'])

        self.ss.clear()
        self.ss.setStartState(ompl_start_scoped)
        self.ss.setGoal(self.goal_region)

        self.ptc = TimeoutOrNotProgressing(self, self.params['termination_criteria'], self.verbose)

        # START TIMING
        t0 = time.time()

        # acutally run the planner
        ob_planner_status = self.ss.solve(self.ptc)

        # END TIMING
        planning_time = time.time() - t0

        # handle results and cleanup
        planner_status = interpret_planner_status(ob_planner_status, self.ptc)

        if planner_status == MyPlannerStatus.Solved:
            ompl_path = self.ss.getSolutionPath()
            actions, planned_path = self.convert_path(ompl_path)
            planner_data = ob.PlannerData(self.si)
            self.rrt.getPlannerData(planner_data)
            tree = planner_data_to_json(planner_data, self.scenario_ompl)
        elif planner_status == MyPlannerStatus.Timeout:
            # Use the approximate solution, since it's usually pretty darn close, and sometimes
            # our goals are impossible to reach so this is important to have
            try:
                ompl_path = self.ss.getSolutionPath()
                actions, planned_path = self.convert_path(ompl_path)
            except RuntimeError:
                rospy.logerr("Timeout before any edges were added. Considering this as Not Progressing.")
                planner_status = MyPlannerStatus.NotProgressing
                actions = []
                tree = {}
                planned_path = [start_state]
            else:  # if no exception was raised
                planner_data = ob.PlannerData(self.si)
                self.rrt.getPlannerData(planner_data)
                tree = planner_data_to_json(planner_data, self.scenario_ompl)
        elif planner_status == MyPlannerStatus.Failure:
            rospy.logerr(f"Failed at starting state: {start_state}")
            tree = {}
            actions = []
            planned_path = [start_state]
        elif planner_status == MyPlannerStatus.NotProgressing:
            tree = {}
            actions = []
            planned_path = [start_state]

        print()
        return PlanningResult(status=planner_status, path=planned_path, actions=actions, time=planning_time, tree=tree)

    def convert_path(self, ompl_path: oc.PathControl) -> Tuple[List[Dict], List[Dict]]:
        planned_path = []
        actions = []
        n_actions = ompl_path.getControlCount()
        for time_idx, state in enumerate(ompl_path.getStates()):
            np_state = self.scenario_ompl.ompl_state_to_numpy(state)
            planned_path.append(np_state)
            if time_idx < n_actions:
                action = ompl_path.getControl(time_idx)
                action_np = self.scenario_ompl.ompl_control_to_numpy(state, action)
                actions.append(action_np)

        return actions, planned_path

    def get_metadata(self):
        return {
            "horizon": self.classifier_model.horizon,
        }


def interpret_planner_status(planner_status: ob.PlannerStatus, ptc: TimeoutOrNotProgressing):
    if str(planner_status) == "Exact solution":
        return MyPlannerStatus.Solved
    elif ptc.not_progressing:
        return MyPlannerStatus.NotProgressing
    elif ptc.timed_out:
        return MyPlannerStatus.Timeout
    else:
        return MyPlannerStatus.Failure
