from dataclasses import dataclass
from enum import Enum
from typing import List, Dict

from link_bot_planning.my_planner import SetupInfo, PlanningQuery


class TrialStatus(Enum):
    Reached = "reached"
    Timeout = "timeout"
    NotProgressingNoRecovery = "not_progressing_no_recovery"


@dataclass
class ExecutionResult:
    path: List[Dict]
    end_trial: bool
    stopped: bool
    end_t: int


@dataclass
class TrialResult:
    setup_info: SetupInfo
    planning_queries: List[PlanningQuery]
    total_time: float
    trial_status: TrialStatus
    trial_idx: int
    goal: Dict
    steps: int
    end_state: Dict


def planning_trial_name(trial_idx: int):
    return f'{trial_idx}_metrics.pkl.gz'