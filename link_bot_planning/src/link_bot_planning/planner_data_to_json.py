from ompl import base as ob
from ompl import control as oc

from link_bot_pycommon.scenario_ompl import ScenarioOmpl
from moonshine.moonshine_utils import listify


def planner_data_to_json(planner_data: oc.PlannerData, scenario_ompl: ScenarioOmpl):
    tree = {
        'vertices': [],
        'edges':    [],
    }

    if planner_data.numVertices() <= 1:
        print("no tree?!")
        return {}

    for vertex_index in range(planner_data.numVertices()):
        v = planner_data.getVertex(vertex_index)
        s = v.getState()
        edges_map = ob.mapUintToPlannerDataEdge()

        np_s = scenario_ompl.ompl_state_to_numpy(s)
        tree['vertices'].append(listify(np_s))

        planner_data.getEdges(vertex_index, edges_map)
        for vertex_index2, control in edges_map:
            control_duration = control.getDuration()
            v2 = planner_data.getVertex(vertex_index2)
            s2 = v2.getState()
            control_np = scenario_ompl.ompl_control_to_numpy(s2, control.getControl())
            np_s2 = scenario_ompl.ompl_state_to_numpy(s2)
            tree['edges'].append(listify({
                'from':     np_s,
                'action':   control_np,
                'duration': control_duration,
                'to':       np_s2,
            }))
    return tree
