from link_bot_pycommon.dual_arm_rope_with_robot_feasibility_checking_scenario import \
    DualArmRopeWithRobotFeasibilityCheckingScenario
from link_bot_pycommon.dual_arm_sim_rope_scenario import SimValDualArmRopeScenario
from tf.transformations import quaternion_from_euler


class DualArmRopeSimValWithRobotFeasibilityCheckingScenario(DualArmRopeWithRobotFeasibilityCheckingScenario,
                                                            SimValDualArmRopeScenario):
    def __init__(self):
        SimValDualArmRopeScenario.__init__(self)
        DualArmRopeWithRobotFeasibilityCheckingScenario.__init__(self, self.robot_namespace)
