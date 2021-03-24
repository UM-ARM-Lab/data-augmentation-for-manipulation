from link_bot_pycommon.dual_arm_rope_with_robot_feasibility_checking_scenario import \
    DualArmRopeWithRobotFeasibilityCheckingScenario
from link_bot_pycommon.dual_arm_sim_rope_scenario import SimValDualArmRopeScenario
from tf.transformations import quaternion_from_euler


class DualArmRopeSimValWithRobotFeasibilityCheckingScenario(DualArmRopeWithRobotFeasibilityCheckingScenario,
                                                            SimValDualArmRopeScenario):
    def __init__(self):
        SimValDualArmRopeScenario.__init__(self)
        DualArmRopeWithRobotFeasibilityCheckingScenario.__init__(self, self.robot_namespace)

        self.left_preferred_tool_orientation = quaternion_from_euler(3.054, -0.851, 0.98)
        self.right_preferred_tool_orientation = quaternion_from_euler(2.254, -0.747, 3.000)
