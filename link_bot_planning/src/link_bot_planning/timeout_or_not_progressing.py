from time import perf_counter
from typing import Dict

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    import ompl.base as ob

import rospy


class TimeoutOrNotProgressing(ob.PlannerTerminationCondition):
    def __init__(self, planner, params: Dict, verbose: int):
        super().__init__(ob.PlannerTerminationConditionFn(self.condition))
        self.params = params
        self.planner = planner
        self.verbose = verbose
        self.t0 = perf_counter()
        self.attempted_extensions = 0
        self.all_rejected = True
        self.not_progressing = None
        self.timed_out = False

    def condition(self):
        self.not_progressing = self.attempted_extensions >= self.params['attempted_extensions_threshold'] and self.all_rejected
        now = perf_counter()
        dt_s = now - self.t0
        self.timed_out = dt_s > self.params['timeout']
        should_terminate = self.timed_out or self.not_progressing
        if self.verbose >= 3:
            msg = f"PTC: {dt_s:.1f}s/{self.params['timeout']:.1f}s {self.attempted_extensions:5d}, {self.all_rejected}"
            rospy.loginfo(msg)
        return should_terminate
