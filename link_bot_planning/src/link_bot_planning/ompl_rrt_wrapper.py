import time
import warnings
from typing import Dict, List, Tuple

from arc_utilities.algorithms import zip_repeat_shorter
from link_bot_pycommon.spinners import SynchronousSpinner

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    import ompl.base as ob
    import ompl.control as oc

import numpy as np
import tensorflow as tf
from matplotlib import cm

from link_bot_planning.get_ompl_scenario import get_ompl_scenario
from link_bot_planning.my_planner import MyPlannerStatus, PlanningQuery, PlanningResult, MyPlanner, LoggingTree, \
    SharedPlanningStateOMPL
from link_bot_planning.trajectory_optimizer import TrajectoryOptimizer
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from state_space_dynamics.base_filter_function import BaseFilterFunction

import rospy
from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_planning.timeout_or_not_progressing import TimeoutOrNotProgressing
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


def add_accept_probability(accept_probabilities, final_predicted_state):
    if 'NNClassifierWrapper' in accept_probabilities:
        final_predicted_state['accept_probability'] = accept_probabilities['NNClassifierWrapper']
    elif 'NNClassifier2Wrapper' in accept_probabilities:
        final_predicted_state['accept_probability'] = accept_probabilities['NNClassifier2Wrapper']
    else:
        final_predicted_state['accept_probability'] = -1


class OmplRRTWrapper(MyPlanner):

    def __init__(self,
                 fwd_model: BaseDynamicsFunction,
                 filter_model: BaseFilterFunction,
                 classifier_models: List[BaseConstraintChecker],
                 planner_params: Dict,
                 action_params: Dict,
                 scenario: ScenarioWithVisualization,
                 verbose: int,
                 log_full_tree: bool = True,
                 ):
        super().__init__(scenario=scenario, fwd_model=fwd_model, filter_model=filter_model)
        self.log_full_tree = log_full_tree
        self.verbose = verbose
        self.fwd_model = fwd_model
        self.classifier_models = classifier_models
        self.params = planner_params
        self.action_params = action_params
        # These RNGs get re-seeded before planning, don't bother changing these values
        self.accept_rng = np.random.RandomState(0)
        self.state_sampler_rng = np.random.RandomState(0)
        self.goal_sampler_rng = np.random.RandomState(0)
        self.control_sampler_rng = np.random.RandomState(0)
        self.scenario = scenario
        self.sps = SharedPlanningStateOMPL()
        self.scenario_ompl = get_ompl_scenario(self.scenario,
                                               planner_params=self.params,
                                               action_params=self.action_params,
                                               state_sampler_rng=self.state_sampler_rng,
                                               control_sampler_rng=self.control_sampler_rng,
                                               shared_planning_state=self.sps,
                                               plot=self.verbose >= 2)

        self.ss = oc.SimpleSetup(self.scenario_ompl.control_space)

        self.si: oc.SpaceInformation = self.ss.getSpaceInformation()

        def _local_planner_cost_function(actions: List[Dict],
                                         environment: Dict,
                                         goal_state: Dict,
                                         states: List[Dict]):
            assert goal_state is None
            goal_cost = self.scenario.distance_to_goal(state=states[1], goal=self.goal_region.goal)
            action_cost = self.scenario.actions_cost(states, actions, self.action_params)
            return goal_cost * self.params['goal_alpha'] + action_cost * self.params['action_alpha']

        self.opt = TrajectoryOptimizer(fwd_model=self.fwd_model,
                                       classifier_model=None,
                                       scenario=self.scenario,
                                       params=self.params,
                                       verbose=self.verbose,
                                       cost_function=_local_planner_cost_function)

        if self.params['use_local_planner']:
            def _dcs_allocator(si):
                return self.scenario_ompl.make_directed_control_sampler(si,
                                                                        rng=self.control_sampler_rng,
                                                                        action_params=action_params,
                                                                        opt=self.opt,
                                                                        max_steps=self.params.get('max_steps', 50))

            self.si.setDirectedControlSamplerAllocator(oc.DirectedControlSamplerAllocator(_dcs_allocator))
        else:
            rospy.loginfo("No DCS, falling back to CS")

        self.ss.setStatePropagator(oc.AdvancedStatePropagatorFn(self.propagate))
        self.ss.setMotionsValidityChecker(oc.MotionsValidityCheckerFn(self.motions_valid))
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_valid))

        self.cleanup_before_plan(0)

        self.rrt = oc.RRT(self.si)
        self.rrt.setIntermediateStates(True)  # this is necessary, because we use this to generate datasets

        self.ss.setPlanner(self.rrt)
        self.si.setMinMaxControlDuration(1, self.params.get('max_steps', 50))

        self.visualize_propogation_color = [0, 0, 0]

    def cleanup_before_plan(self, seed):
        self.tree = LoggingTree()
        self.ptc = None
        self.n_total_action = None
        self.goal_region = None
        # a Dictionary containing the parts of state which are not predicted/planned for, i.e. the environment
        self.start_state = None
        self.closest_state_to_goal = None
        self.min_dist_to_goal = 10000

        self.accept_rng.seed(seed)
        self.state_sampler_rng.seed(seed)
        self.goal_sampler_rng.seed(seed)
        self.control_sampler_rng.seed(seed)

        self.last_propagate_time = time.perf_counter()
        self.progagate_dts = []

    def is_valid(self, state):
        valid = self.scenario_ompl.state_space.satisfiesBounds(state)
        return valid

    def motions_valid(self, motions):
        final_state = self.scenario_ompl.ompl_state_to_numpy(motions[-1].getState())

        motions_valid = final_state['num_diverged'] < self.params['horizon'] - 1  # yes, minus 1
        motions_valid = bool(np.squeeze(motions_valid))
        if not motions_valid:
            if self.verbose >= 2:
                self.scenario.plot_rejected_state(final_state)

        # PTC bookkeeping to figure out how the planner is progressing
        self.ptc.attempted_extensions += 1
        if motions_valid:
            self.ptc.all_rejected = False
            if self.goal_region.canSample():
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
            # skip the first (null) action, because that would represent the action that brings us to the first state
            if t > 0:
                actions.append(self.scenario_ompl.ompl_control_to_numpy(state, control))
        actions = np.array(actions)
        return states_sequence, actions

    def predict(self, previous_states, previous_actions, new_action):
        """
        Here we not only use the forward model to predict the future states, but we also feed the transition through
        the classifier and update the num_diverged state component.
        Args:
            previous_states:
            previous_actions:
            new_action:

        Returns:

        """
        new_actions = [new_action]
        last_previous_state = previous_states[-1]
        mean_predicted_states, stdev_predicted_states = self.fwd_model.propagate(environment=self.sps.environment,
                                                                                 start_state=last_previous_state,
                                                                                 actions=new_actions)
        for t, dict_of_stdevs in enumerate(stdev_predicted_states):
            mean_predicted_states[t]['stdev'] = np.sum(np.concatenate(list(dict_of_stdevs.values())), keepdims=True)
        # get only the final state predicted, since *_predicted_states includes the start state
        final_predicted_state = mean_predicted_states[-1]

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

        # compute new num_diverged by checking the constraint
        accept, accept_probabilities = self.check_constraint(all_states, all_actions)
        add_accept_probability(accept_probabilities, final_predicted_state)

        if accept:
            final_predicted_state['num_diverged'] = np.array([0.0])

        else:
            final_predicted_state['num_diverged'] = last_previous_state['num_diverged'] + 1

        return final_predicted_state, accept, accept_probabilities

    def check_constraint(self, states: List[Dict], actions: List[Dict]):
        accept = True
        accept_probabilities = {}
        for classifier in self.classifier_models:
            p_accepts_for_model = classifier.check_constraint(environment=self.sps.environment,
                                                              states_sequence=states,
                                                              actions=actions)

            assert p_accepts_for_model.ndim == 1

            accept_probabilities[classifier.name] = p_accepts_for_model

            # NOTE: Here is where we decide whether to accept a transition or not.
            #  you could do this with a simple threshold, or by saying p(accept) is a function of classifier output
            accept_type = self.params.get('accept_type', 'strict')
            if accept_type == 'strict':
                accepts = p_accepts_for_model > self.params['accept_threshold']
                accept = np.all(accepts)
                if not accept:
                    break
            elif accept_type == 'probabilistic':
                # https://arxiv.org/pdf/2001.11051.pdf, see Algorithm 1
                accepts = p_accepts_for_model > self.params['accept_threshold']
                accept = np.all(accepts)
                if not accept:
                    r = self.accept_rng.uniform(0, 1)
                    accept = (r < self.params['probabilistic_accept_k'])
                    if not accept:
                        break
            else:
                raise NotImplementedError(f"invalid {accept_type:=}")

        return accept, accept_probabilities

    def propagate(self, motions, control, duration, state_out):
        del duration  # unused, multi-step propagation is handled inside propagateMotionsWhileValid

        # measure performance
        now = time.perf_counter()
        dt = now - self.last_propagate_time
        self.progagate_dts.append(dt)
        self.last_propagate_time = now

        self.spinner.update()

        # Convert from OMPL -> Numpy
        previous_states, previous_actions = self.motions_to_numpy(motions)
        previous_state = previous_states[-1]
        previous_ompl_state = motions[-1].getState()
        new_action = self.scenario_ompl.ompl_control_to_numpy(previous_ompl_state, control)
        np_final_state, accept, accept_probabilities = self.predict(previous_states,
                                                                    previous_actions,
                                                                    new_action)

        # Convert back Numpy -> OMPL
        self.scenario_ompl.numpy_to_ompl_state(np_final_state, state_out)

        # log the data
        if self.log_full_tree:
            self.tree.add(before_state=previous_state,
                          action=new_action,
                          after_state=np_final_state,
                          accept_probabilities=accept_probabilities)

        self.scenario.heartbeat()

        # visualize
        if self.verbose >= 2:
            self.visualize_propogation(accept,
                                       accept_probabilities,
                                       new_action,
                                       np_final_state,
                                       previous_actions,
                                       previous_state,
                                       state_out)

    def visualize_propogation(self,
                              accept: bool,
                              accept_probabilities: Dict,
                              new_action: Dict,
                              np_final_state: Dict,
                              previous_actions: Dict,
                              previous_state: Dict,
                              state_out: ob.CompoundState):
        # check if this is a new action, in which case we want to sample a new color
        if self.sps.just_sampled_new_action:
            self.sps.just_sampled_new_action = False
            random_color = cm.Dark2(self.control_sampler_rng.uniform(0, 1))
            self.visualize_propogation_color = random_color

        classifier_probabilities = None
        if 'NNClassifierWrapper' in accept_probabilities:
            classifier_probabilities = accept_probabilities['NNClassifierWrapper']
        if 'NNClassifier2Wrapper' in accept_probabilities:
            classifier_probabilities = accept_probabilities['NNClassifier2Wrapper']
        if classifier_probabilities is not None:
            assert classifier_probabilities.size == 1
            classifier_probability = classifier_probabilities[0]
            alpha = min(classifier_probability * 0.8 + 0.2, 0.8)
            classifier_probability_color = cm.Reds_r(classifier_probability)
            self.scenario.plot_accept_probability(classifier_probability)
        else:
            alpha = 0.8
            classifier_probability_color = cm.Reds_r(1.0)

        statisfies_bounds = self.scenario_ompl.state_space.satisfiesBounds(state_out)
        if accept and statisfies_bounds:
            self.scenario.plot_tree_state(np_final_state, color=classifier_probability_color)
            self.scenario.plot_tree_action(previous_state,
                                           new_action,
                                           r=self.visualize_propogation_color[0],
                                           g=self.visualize_propogation_color[1],
                                           b=self.visualize_propogation_color[2],
                                           a=alpha)
        else:
            self.scenario.plot_rejected_state(np_final_state)

        self.scenario.plot_current_tree_state(np_final_state, color=classifier_probability_color)
        self.scenario.plot_current_tree_action(previous_state, new_action,
                                               r=self.visualize_propogation_color[0],
                                               g=self.visualize_propogation_color[1],
                                               b=self.visualize_propogation_color[2],
                                               a=alpha)

    def make_goal_region(self, goal):
        return self.scenario_ompl.make_goal_region(self.si,
                                                   rng=self.goal_sampler_rng,
                                                   params=self.params,
                                                   goal=goal,
                                                   plot=self.verbose >= 2)

    def make_ptc(self, planning_query: PlanningQuery):
        return TimeoutOrNotProgressing(planning_query, self.params['termination_criteria'], self.verbose)

    def plan(self, planning_query: PlanningQuery):
        self.spinner = SynchronousSpinner('Planning')
        self.cleanup_before_plan(planning_query.seed)

        self.sps.environment = planning_query.environment

        self.goal_region = self.make_goal_region(goal=planning_query.goal)

        # create start and goal states
        start_state = planning_query.start
        start_state['stdev'] = np.array([0.0])
        start_state['num_diverged'] = np.array([0.0])
        self.start_state = start_state
        ompl_start_scoped = ob.State(self.scenario_ompl.state_space)
        self.scenario_ompl.numpy_to_ompl_state(start_state, ompl_start_scoped())

        # visualization
        self.scenario.reset_planning_viz()
        self.scenario.plot_environment_rviz(planning_query.environment)
        self.scenario.plot_start_state(start_state)
        self.scenario.plot_goal_rviz(planning_query.goal, self.params['goal_params']['threshold'])

        self.ss.clear()
        self.ss.setStartState(ompl_start_scoped)
        self.ss.setGoal(self.goal_region)

        self.ptc = self.make_ptc(planning_query)

        # START TIMING
        t0 = time.time()

        # actually run the planner
        ob_planner_status = self.ss.solve(self.ptc)

        # END TIMING
        planning_time = time.time() - t0

        mean_propagate_time = float(np.mean(self.progagate_dts))
        if self.verbose >= 1:
            print(f"\nMean Propagate Time = {mean_propagate_time:.4f}s")

        # handle results and cleanup
        planner_status = self.ptc.interpret_planner_status(ob_planner_status)

        if planner_status == MyPlannerStatus.Solved:
            ompl_path = self.ss.getSolutionPath()
            actions, planned_path = self.convert_path(ompl_path)
            if self.params['smooth']:
                actions, planned_path = self.smooth(planning_query, actions, planned_path)
        elif planner_status == MyPlannerStatus.Timeout:
            # Use the approximate solution, since it's usually pretty darn close, and sometimes
            # our goals are impossible to reach so this is important to have
            try:
                ompl_path = self.ss.getSolutionPath()
                actions, planned_path = self.convert_path(ompl_path)
                if self.params['smooth']:
                    actions, planned_path = self.smooth(planning_query, actions, planned_path)
            except RuntimeError:
                rospy.logerr("Timeout before any edges were added. Considering this as Not Progressing.")
                planner_status = MyPlannerStatus.NotProgressing
                actions = []
                planned_path = [start_state]
        elif planner_status == MyPlannerStatus.Failure:
            rospy.logerr(f"Failed at starting state: {start_state}")
            actions = []
            planned_path = [start_state]
        elif planner_status == MyPlannerStatus.NotProgressing:
            actions = []
            planned_path = [start_state]
        else:
            raise ValueError(f"invalid planner status {planner_status}")

        if self.verbose >= 0:
            self.plot_path(state_sequence=planned_path, action_sequence=actions)

        self.spinner.stop()
        return PlanningResult(status=planner_status,
                              path=planned_path,
                              actions=actions,
                              time=planning_time,
                              mean_propagate_time=mean_propagate_time,
                              tree=self.tree)

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
            "horizon": self.classifier_models[0].horizon,
        }

    def smooth(self, planning_query: PlanningQuery, action_sequence: List[Dict], state_sequence: List[Dict]):
        env = planning_query.environment
        goal = planning_query.goal
        initial_final_state = state_sequence[-1]
        initial_distance_to_goal = self.scenario.distance_to_goal(initial_final_state, goal)

        if self.verbose >= 2:
            self.plot_path(state_sequence=state_sequence, action_sequence=action_sequence)

        smoothing_rng = np.random.RandomState(0)
        n_shortcut_attempts = self.params.get('n_shortcut_attempts', 50)
        t0 = time.perf_counter()
        for j in range(n_shortcut_attempts):

            plan_length = len(state_sequence)
            if plan_length < 4:
                return action_sequence, state_sequence

            # randomly sample a start index
            start_t = smoothing_rng.randint(0, plan_length - 3)

            # sample an end index
            max_shortcut_length = self.params.get('max_shortcut_length', 20)  # too long and we run out of GPU memory
            shortcut_max_t = min(start_t + max_shortcut_length, plan_length - 1)
            shortcut_min_t = start_t + 2
            if shortcut_min_t >= shortcut_max_t:
                rospy.logerr(f"smoothing sampling bug?! {start_t=}, {plan_length=}")
                continue
            end_t = smoothing_rng.randint(shortcut_min_t, shortcut_max_t)

            # interpolate the grippers to make a new action sequence
            start_state = state_sequence[start_t]
            end_state = state_sequence[end_t]
            shortcut_action_seq = self.scenario.interpolate(start_state, end_state)
            # these actions need to be re-propagated
            proposed_action_seq_to_end = shortcut_action_seq + action_sequence[end_t:]
            proposed_action_seq = action_sequence[:start_t] + proposed_action_seq_to_end

            proposed_state_seq_to_end, _ = self.fwd_model.propagate(env,
                                                                    state_sequence[start_t],
                                                                    proposed_action_seq_to_end)
            classifier_accept, accept_probabilities = self.check_constraint(proposed_state_seq_to_end,
                                                                            proposed_action_seq_to_end)
            # copy the old/new accept probabilites in the states, cuz propagate produces state w/o accept probabilities
            proposed_state_seq_to_end[0]['accept_probability'] = state_sequence[start_t]['accept_probability']
            for j, proposed_state_j in enumerate(proposed_state_seq_to_end[1:]):
                if 'NNClassifierWrapper' in accept_probabilities:
                    proposed_state_j['accept_probability'] = np.array([accept_probabilities['NNClassifierWrapper'][j]], np.float32)
                elif 'NNClassifier2Wrapper' in accept_probabilities:
                    proposed_state_j['accept_probability'] = np.array([accept_probabilities['NNClassifier2Wrapper'][j]], np.float32)
                else:
                    proposed_state_j['accept_probability'] = np.array([-1], np.float32)
            proposed_state_seq = state_sequence[:start_t] + proposed_state_seq_to_end

            # NOTE: we don't check this because smoothing is run even when we Timeout and the goal wasn't reached
            distance_to_goal = self.scenario.distance_to_goal(proposed_state_seq[-1], goal)
            much_further_from_goal = distance_to_goal - initial_distance_to_goal > 0.03  # FIXME: hardcoded parameter
            if classifier_accept and much_further_from_goal:
                rospy.logwarn("smoothing would have made distance to goal higher")

            # if the shortcut was successful, save that as the new path
            accept = tf.logical_and(classifier_accept, tf.logical_not(much_further_from_goal))
            if accept:
                state_sequence = proposed_state_seq
                action_sequence = proposed_action_seq

                # take the current planned path and add it to the logging tree
                for state, action, next_state in zip(state_sequence[:-1], action_sequence, state_sequence[1:]):
                    if self.log_full_tree:
                        self.tree.add(before_state=state,
                                      action=action,
                                      after_state=next_state,
                                      accept_probabilities=accept_probabilities)

                # visualize & debug info
                if self.verbose >= 3:
                    self.clear_smoothing_markers()
                    self.scenario.plot_state_rviz(start_state, idx=0, label='from', color='y')
                    self.scenario.plot_state_rviz(end_state, idx=1, label='to', color='m')
                    self.plot_path(state_sequence=state_sequence, action_sequence=action_sequence)
                if self.verbose >= 1:
                    print(f"shortcut from {start_t} to {end_t} accepted. Plan length is now {plan_length}")

        # Plot the smoothed result
        if self.verbose >= 2:
            self.plot_path(state_sequence=state_sequence, action_sequence=action_sequence)

        dt = time.perf_counter() - t0
        if self.verbose >= 0:
            print(f"Smoothing Time = {dt:.3f}s")

        return action_sequence, state_sequence

    def plot_path(self, state_sequence, action_sequence, label='smoothed'):
        if len(action_sequence) == 0 or len(state_sequence) == 0:
            return

        # TODO: make this one message so dropped messages are less of an issue?
        for t, (state_t, action_t) in enumerate(zip_repeat_shorter(state_sequence, action_sequence)):
            self.scenario.plot_state_rviz(state_t, label=label, idx=t)
            if action_t:
                self.scenario.plot_action_rviz(state_t, action_t, label=label, idx=t)
            time.sleep(0.1)

    def clear_smoothing_markers(self):
        # FIXME: temporary hack
        self.scenario.reset_planning_viz()
        # self.scenario.mm.delete(label='from')
        # self.scenario.mm.delete(label='to')
        # self.scenario.mm.delete(label='smoothed')
