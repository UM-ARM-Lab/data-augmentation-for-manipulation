from link_bot_planning.dual_arm_rope_ompl import DualArmRopeOmpl
from link_bot_planning.floating_rope_ompl import FloatingRopeOmpl
from link_bot_planning.rope_dragging_ompl import RopeDraggingOmpl
from link_bot_pycommon.dual_arm_sim_rope_scenario import SimDualArmRopeScenario
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.floating_rope_scenario import FloatingRopeScenario
from link_bot_pycommon.rope_dragging_scenario import RopeDraggingScenario
from link_bot_pycommon.scenario_ompl import ScenarioOmpl


def get_ompl_scenario(scenario: ExperimentScenario, *args, **kwargs) -> ScenarioOmpl:
    # order matters here, because of inheritence
    if isinstance(scenario, RopeDraggingScenario):
        return RopeDraggingOmpl(scenario, *args, **kwargs)
    elif isinstance(scenario, SimDualArmRopeScenario):
        return DualArmRopeOmpl(scenario, *args, **kwargs)
    elif isinstance(scenario, FloatingRopeScenario):
        return FloatingRopeOmpl(scenario, *args, **kwargs)
    else:
        raise NotImplementedError(f"unimplemented scenario {scenario}")
