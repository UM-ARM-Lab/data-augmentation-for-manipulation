from functools import lru_cache
from typing import Optional

from colorama import Fore


def make_toy_scenario():
    from dm_envs.toy_example_scenario import ToyExampleScenario
    return ToyExampleScenario


def make_rope_dragging_scenario():
    from link_bot_pycommon.rope_dragging_scenario import RopeDraggingScenario
    return RopeDraggingScenario


def make_dual_arm_real_victor_rope_scenario():
    from link_bot_pycommon.dual_arm_real_victor_rope_scenario import DualArmRealVictorRopeScenario
    return DualArmRealVictorRopeScenario


def make_dual_arm_real_val_rope_scenario():
    from link_bot_pycommon.dual_arm_real_val_rope_scenario import DualArmRealValRopeScenario
    return DualArmRealValRopeScenario


def make_cylinders_scenario():
    from dm_envs.cylinders_scenario import CylindersScenario
    return CylindersScenario


def make_blocks_scenario():
    from dm_envs.blocks_scenario import BlocksScenario
    return BlocksScenario


def make_dual_arm_scenario():
    from link_bot_pycommon.dual_arm_scenario import DualArmScenario
    return DualArmScenario


def make_dual_arm_sim_victor_scenario():
    from link_bot_pycommon.dual_arm_sim_rope_scenario import SimVictorDualArmRopeScenario
    return SimVictorDualArmRopeScenario


def make_dual_arm_sim_val_scenario():
    from link_bot_pycommon.dual_arm_sim_rope_scenario import SimValDualArmRopeScenario
    return SimValDualArmRopeScenario


def make_floating_rope_scenario():
    from link_bot_pycommon.floating_rope_scenario import FloatingRopeScenario
    return FloatingRopeScenario


def make_dual_arm_rope_sim_val_with_robot_feasibility_checking():
    from link_bot_pycommon.with_robot_feasibility_checking_scenario import \
        DualArmRopeSimValWithRobotFeasibilityCheckingScenario
    return DualArmRopeSimValWithRobotFeasibilityCheckingScenario


def make_real_val_with_robot_feasibility_checking():
    from link_bot_pycommon.with_robot_feasibility_checking_scenario import \
        DualArmRopeRealValWithRobotFeasibilityCheckingScenario
    return DualArmRopeRealValWithRobotFeasibilityCheckingScenario


scenario_map = {
    'link_bot':                                              make_rope_dragging_scenario,
    'rope dragging':                                         make_rope_dragging_scenario,
    'rope_dragging':                                         make_rope_dragging_scenario,
    'dragging':                                              make_rope_dragging_scenario,
    'dual_arm':                                              make_dual_arm_sim_victor_scenario,
    'dual_arm_real_victor':                                  make_dual_arm_real_victor_rope_scenario,
    'dual_arm_real_val':                                     make_dual_arm_real_val_rope_scenario,
    'dual_arm_rope_sim_victor':                              make_dual_arm_sim_victor_scenario,
    'dual_arm_rope_sim_val':                                 make_dual_arm_sim_val_scenario,
    'dual_arm_rope':                                         make_dual_arm_sim_victor_scenario,
    'dual_floating_gripper_rope':                            make_floating_rope_scenario,
    'dual_floating':                                         make_floating_rope_scenario,
    'floating_rope':                                         make_floating_rope_scenario,
    'dual_arm_no_rope':                                      make_dual_arm_scenario,
    'dual_arm_rope_sim_val_with_robot_feasibility_checking': make_dual_arm_rope_sim_val_with_robot_feasibility_checking,
    'real_val_with_robot_feasibility_checking':              make_real_val_with_robot_feasibility_checking,
    'blocks':                                                make_blocks_scenario,
    'cylinders':                                             make_cylinders_scenario,
    'toy':                                                   make_toy_scenario,
}


def get_scenario(scenario_name: str, params=None):
    """ With this approach, we only ever import the scenario we want to use. Nice! """
    if params is None:
        params = {}
    if scenario_name == 'dual_arm':
        print(Fore.YELLOW + "Please update the scenario name! dual_arm is deprecated because it's not specific enough")
    if scenario_name not in scenario_map:
        raise NotImplementedError(scenario_name)
    return scenario_map[scenario_name]()(params)


@lru_cache
def get_scenario_cached(scenario_name: str):
    return get_scenario(scenario_name)
