from typing import Dict

from link_bot_pycommon.base_dual_arm_rope_scenario import BaseDualArmRopeScenario
from moonshine.moonshine_utils import add_batch, remove_batch


class DualArmRopeWithRobotFeasibilityCheckingScenario:

    def __init__(self, robot_namespace: str):
        self.robot_namespace = robot_namespace

    def is_action_valid(self, environment: Dict, state: Dict, action: Dict, action_params: Dict):
        valid = BaseDualArmRopeScenario.is_action_valid(self, environment, state, action, action_params)
        if not valid:
            return False

        # further checks for if the motion is feasible under the controller
        example = {}
        # add batch dimension
        example.update(add_batch(environment))
        # add batch and time dimensions
        example.update(add_batch(add_batch(state)))
        example.update(add_batch(add_batch(action)))
        example['batch_size'] = 1
        target_reached, _, __ = BaseDualArmRopeScenario.follow_jacobian_from_example(self, example)
        target_reached = remove_batch(target_reached)[1]  # t=1, target reached for t=0 is always true

        return target_reached
