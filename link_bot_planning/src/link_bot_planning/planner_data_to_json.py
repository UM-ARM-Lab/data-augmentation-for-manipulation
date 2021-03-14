from ompl import base as ob
from ompl import control as oc

from link_bot_planning.my_planner import LoggingTree
from link_bot_pycommon.scenario_ompl import ScenarioOmpl
from moonshine.moonshine_utils import listify


def planner_data_to_json(planner_data: oc.PlannerData, scenario_ompl: ScenarioOmpl):
    tree = LoggingTree()

    if planner_data.numVertices() <= 1:
        print("no tree?!")
        return {}

    for before_state_idx in range(planner_data.numVertices()):
        v = planner_data.getVertex(before_state_idx)
        s = v.getState()
        edges_map = ob.mapUintToPlannerDataEdge()

        before_state = scenario_ompl.ompl_state_to_numpy(s)

        planner_data.getEdges(before_state_idx, edges_map)
        for after_state_idx, control in edges_map:
            v2 = planner_data.getVertex(after_state_idx)
            s2 = v2.getState()
            action = scenario_ompl.ompl_control_to_numpy(s2, control.getControl())
            after_state = scenario_ompl.ompl_state_to_numpy(s2)

            before_state['ompl_vertex_idx'] = before_state_idx
            after_state['ompl_vertex_idx'] = after_state_idx
            tree.add(before_state, action, after_state)
    return tree
