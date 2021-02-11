from ompl import base as ob

from link_bot_pycommon.scenario_ompl import ScenarioOmpl
from moonshine.moonshine_utils import listify


def planner_data_to_json(planner_data, scenario_ompl: ScenarioOmpl):
    json = {
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
        json['vertices'].append(listify(np_s))

        planner_data.getEdges(vertex_index, edges_map)
        for vertex_index2 in edges_map.keys():
            v2 = planner_data.getVertex(vertex_index2)
            s2 = v2.getState()
            np_s2 = scenario_ompl.ompl_state_to_numpy(s2)
            json['edges'].append(listify({
                'from': np_s,
                'to':   np_s2,
            }))
    return json
