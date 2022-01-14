from typing import Dict

from link_bot_pycommon.base_dual_arm_rope_scenario import BaseDualArmRopeScenario
from link_bot_pycommon.dual_arm_real_val_rope_scenario import DualArmRealValRopeScenario
from link_bot_pycommon.dual_arm_rope_with_robot_feasibility_checking_scenario import \
    DualArmRopeWithRobotFeasibilityCheckingScenario
from link_bot_pycommon.dual_arm_sim_rope_scenario import SimValDualArmRopeScenario


class DualArmRopeSimValWithRobotFeasibilityCheckingScenario(DualArmRopeWithRobotFeasibilityCheckingScenario,
                                                            SimValDualArmRopeScenario):
    def __init__(self):
        SimValDualArmRopeScenario.__init__(self)
        DualArmRopeWithRobotFeasibilityCheckingScenario.__init__(self, self.robot_namespace)

    def simple_name(self):
        return "dual_arm_rope_sim_val_with_robot_feasibility_checking"

    def __repr__(self):
        return self.simple_name()


class DualArmRopeRealValWithRobotFeasibilityCheckingScenario(DualArmRopeWithRobotFeasibilityCheckingScenario,
                                                             DualArmRealValRopeScenario):
    def __init__(self):
        DualArmRealValRopeScenario.__init__(self)
        DualArmRopeWithRobotFeasibilityCheckingScenario.__init__(self, self.robot_namespace)

    def simple_name(self):
        return "real_val_with_robot_feasibility_checking"

    def is_action_valid(self, environment: Dict, state: Dict, action: Dict, action_params: Dict):
        valid = BaseDualArmRopeScenario.is_action_valid(self, environment, state, action, action_params)
        if not valid:
            return False

        action_fk = self.action_relative_to_fk(action, state)
        target_reached = self.is_motion_feasible(action_fk, environment, state)

        return target_reached


    def __repr__(self):
        return self.simple_name()
