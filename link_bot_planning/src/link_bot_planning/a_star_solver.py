from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from python_astar.astar import AStar


class AStarSolver(AStar):
    def __init__(self, scenario: ScenarioWithVisualization, tree, goal_threshold):
        self.scenario = scenario
        self.tree = tree
        self.goal_threshold = goal_threshold

    def heuristic_cost_estimate(self, n1, n2):
        return 0

    def distance_between(self, n1, n2):
        return 1

    def is_goal_reached(self, current, goal):
        return self.scenario.distance_to_any_point_goal(current, goal).numpy() < self.goal_threshold

    def neighbors(self, node):
        # return a list of nodes
        return self.tree[node]

    def viz_current(self, s):
        self.scenario.plot_state_rviz(s, label='current')

    def viz_neighbor_visited(self, s):
        self.scenario.plot_state_rviz(s, label='neighbor_visited')
