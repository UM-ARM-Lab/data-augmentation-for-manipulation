import warnings
from time import perf_counter
from typing import Dict

from link_bot_planning.my_planner import PlanningQuery

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
