import warnings
from time import perf_counter
from typing import Dict

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
        self.not_progressing = None
        self.timed_out = False
        self.threshold = self.params['attempted_extensions_threshold']
        self.timeout = self.params['timeout']
        self.all_rejected = True
        self.dt_s = 0

    def condition(self):
        self.not_progressing = self.attempted_extensions >= self.threshold and self.all_rejected
        now = perf_counter()
        self.dt_s = now - self.t0
        self.timed_out = self.dt_s > self.timeout
        should_terminate = self.timed_out or self.not_progressing
        if self.verbose >= 3:
            rospy.loginfo(self.status_msg())
        return should_terminate

    def status_msg(self):
        msg = f"{self.dt_s:.1f}s/{self.timeout:.1f}s {self.attempted_extensions:5d}, {self.all_rejected}"
        return "PTC: " + msg
