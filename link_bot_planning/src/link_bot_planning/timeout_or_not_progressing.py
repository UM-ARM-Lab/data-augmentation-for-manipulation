import warnings
from time import perf_counter
from typing import Dict

from link_bot_planning.my_planner import PlanningQuery, MyPlannerStatus

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    import ompl.base as ob


class TimeoutOrNotProgressing(ob.PlannerTerminationCondition):
    def __init__(self, planning_query: PlanningQuery, params: Dict, verbose: int):
        super().__init__(ob.PlannerTerminationConditionFn(self.condition))
        self.params = params
        self.verbose = verbose
        self.threshold = self.params['attempted_extensions_threshold']
        self.timeout = self.params['timeout']
        self.total_timeout = self.params['total_timeout']
        self.planning_query = planning_query
        self.start_time = planning_query.trial_start_time_seconds

        self.all_rejected = True
        self.not_progressing = None
        self.timed_out = False
        self.attempted_extensions = 0
        self.debugging_terminate = False

        self.t0 = perf_counter()

    def condition(self):
        self.not_progressing = self.attempted_extensions >= self.threshold and self.all_rejected
        now = perf_counter()
        dt_s = now - self.t0
        total_trial_dt_s = now - self.start_time
        planning_query_timed_out = dt_s > self.timeout
        total_trial_timed_out = total_trial_dt_s > self.total_timeout
        self.timed_out = planning_query_timed_out or total_trial_timed_out
        should_terminate = self.timed_out or self.not_progressing

        return should_terminate

    def interpret_planner_status(self, planner_status: ob.PlannerStatus):
        if str(planner_status) == "Exact solution":
            return MyPlannerStatus.Solved
        elif self.not_progressing:
            return MyPlannerStatus.NotProgressing
        elif self.timed_out:
            return MyPlannerStatus.Timeout
        else:
            return MyPlannerStatus.Failure


class NExtensions(ob.PlannerTerminationCondition):
    def __init__(self, max_n_extensions: int):
        super().__init__(ob.PlannerTerminationConditionFn(self.condition))
        self.max_n_extensions = max_n_extensions
        self.attempted_extensions = 0
        self.not_progressing = False
        self.timed_out = False

    def condition(self):
        should_terminate = self.attempted_extensions >= self.max_n_extensions
        return should_terminate

    def interpret_planner_status(self, _: ob.PlannerStatus):
        return MyPlannerStatus.Timeout


class EvalRecoveryPTC(ob.PlannerTerminationCondition):
    def __init__(self, planning_query: PlanningQuery, params: Dict, verbose: int):
        super().__init__(ob.PlannerTerminationConditionFn(self.condition))
        self.params = params
        self.verbose = verbose
        self.threshold = self.params['attempted_extensions_threshold']
        self.planning_query = planning_query
        self.start_time = planning_query.trial_start_time_seconds

        self.all_rejected = True
        self.not_progressing = None
        self.attempted_extensions = 0
        self.debugging_terminate = False

        self.t0 = perf_counter()

    def condition(self):
        self.not_progressing = self.attempted_extensions >= self.threshold and self.all_rejected
        should_terminate = self.not_progressing or (not self.all_rejected and self.attempted_extensions > 32)  # huh?!

        return should_terminate

    def interpret_planner_status(self, planner_status: ob.PlannerStatus):
        if self.not_progressing:
            return MyPlannerStatus.NotProgressing
        else:
            return MyPlannerStatus.Solved
