import matplotlib.pyplot as plt
import numpy as np
import ompl.base as ob
import ompl.util as ou
from matplotlib.lines import Line2D
from ompl import control as oc

from link_bot_planning.state_spaces import to_numpy, from_numpy


def plot(planner_data, sdf, start, goal, path, controls, n_state, extent):
    plt.figure()
    plt.imshow(np.flipud(sdf.T) > 0, extent=extent)

    print(len(GPDirectedControlSampler.states_sampled_at))
    print(GPDirectedControlSampler.states_sampled_at[0])
    for state_sampled_at in GPDirectedControlSampler.states_sampled_at:
        xs = [state_sampled_at[0, 0], state_sampled_at[0, 2], state_sampled_at[0, 4]]
        ys = [state_sampled_at[0, 1], state_sampled_at[0, 3], state_sampled_at[0, 5]]
        plt.plot(xs, ys, label='sampled states', linewidth=0.5, c='b', alpha=0.5, zorder=1)

    plt.scatter(start[0, 0], start[0, 1], label='start', s=100, c='r', zorder=1)
    plt.scatter(goal[0], goal[1], label='goal', s=100, c='g', zorder=1)
    for path_i in path:
        xs = [path_i[0], path_i[2], path_i[4]]
        ys = [path_i[1], path_i[3], path_i[5]]
        plt.plot(xs, ys, label='final path', linewidth=2, c='cyan', alpha=0.75, zorder=4)
    plt.quiver(path[:-1, 4], path[:-1, 5], controls[:, 0], controls[:, 1], width=0.002, zorder=5, color='k')

    for vertex_index in range(planner_data.numVertices()):
        v = planner_data.getVertex(vertex_index)
        # draw the configuration of the rope
        s = v.getState()
        edges_map = ob.mapUintToPlannerDataEdge()

        np_s = to_numpy(s, n_state)
        plt.scatter(np_s[0, 0], np_s[0, 1], s=15, c='orange', zorder=2, alpha=0.5, label='tail')

        if len(edges_map.keys()) == 0:
            xs = [np_s[0, 0], np_s[0, 2], np_s[0, 4]]
            ys = [np_s[0, 1], np_s[0, 3], np_s[0, 5]]
            plt.plot(xs, ys, linewidth=1, c='orange', alpha=0.2, zorder=2, label='full rope')

        planner_data.getEdges(vertex_index, edges_map)
        for vertex_index2 in edges_map.keys():
            v2 = planner_data.getVertex(vertex_index2)
            s2 = v2.getState()
            np_s2 = to_numpy(s2, n_state)
            plt.plot([np_s[0, 0], np_s2[0, 0]], [np_s[0, 1], np_s2[0, 1]], c='white', label='tree', zorder=1)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.xlim(extent[0:2])
    plt.ylim(extent[2:4])

    custom_lines = [
        Line2D([0], [0], color='b', lw=1),
        Line2D([0], [0], color='r', lw=1),
        Line2D([0], [0], color='g', lw=1),
        Line2D([0], [0], color='cyan', lw=1),
        Line2D([0], [0], color='k', lw=1),
        Line2D([0], [0], color='orange', lw=1),
        Line2D([0], [0], color='orange', lw=1),
        Line2D([0], [0], color='white', lw=1),
    ]

    plt.legend(custom_lines, ['sampled rope configurations', 'start', 'goal', 'final path', 'controls', 'full rope', 'search tree'])


class GPDirectedControlSampler(oc.DirectedControlSampler):
    states_sampled_at = []

    def __init__(self, si, fwd_gp_model, inv_gp_model, max_v):
        super(GPDirectedControlSampler, self).__init__(si)
        self.si = si
        self.name_ = 'gp_dcs'
        self.rng_ = ou.RNG()
        self.max_v = max_v
        self.fwd_gp_model = fwd_gp_model
        self.inv_gp_model = inv_gp_model
        self.state_space = self.si.getStateSpace()
        self.control_space = self.si.getControlSpace()
        self.n_state = self.state_space.getDimension()
        self.n_control = self.control_space.getDimension()
        self.min_steps = int(self.si.getMinControlDuration())
        self.max_steps = int(self.si.getMaxControlDuration())
        self.fwd_gp_model.initialize_rng(self.min_steps, self.max_steps)
        if inv_gp_model is not None:
            self.inv_gp_model.initialize_rng(self.min_steps, self.max_steps)

    @classmethod
    def alloc(cls, si, fwd_gp_model, inv_gp_model, max_v):
        return cls(si, fwd_gp_model, inv_gp_model, max_v)

    @classmethod
    def allocator(cls, fwd_gp_model, inv_gp_model, max_v):
        def partial(si):
            return cls.alloc(si, fwd_gp_model, inv_gp_model, max_v)

        return oc.DirectedControlSamplerAllocator(partial)

    def sampleTo(self, control_out, previous_control, state, target_out):
        np_s = to_numpy(state, self.n_state)
        np_target = to_numpy(target_out, self.n_state)

        # # construct a new np_target which is at most 1 meter away from the rope
        # # 1 meter is ~80th percentile on how far the head moved in the training data for the inverse GP
        # np_s_tail = np_s.reshape(-1, 2)[0]
        # np_target_pts = np_target.reshape(-1, 2)
        # np_target_tail = np_target_pts[0]
        # tail_delta = np_target_tail - np_s_tail
        # max_tail_delta = 1.0
        # new_tail = np_s_tail
        # if np.linalg.norm(tail_delta) > max_tail_delta:
        #     new_tail = np_s_tail + tail_delta / np.linalg.norm(tail_delta) * max_tail_delta
        # new_np_target = (np_target_pts - np_target_tail + new_tail).reshape(-1, self.n_state)
        # np_target = new_np_target

        self.states_sampled_at.append(np_target)

        if self.inv_gp_model is None:
            u = self.fwd_gp_model.dumb_inv_act(np_s, np_target, self.max_v)
        else:
            u = self.inv_gp_model.inv_act(np_s, np_target, self.max_v)

        np_s_next = self.fwd_gp_model.fwd_act(np_s, u)

        from_numpy(u, control_out, self.n_control)
        from_numpy(np_s_next, target_out, self.n_state)

        # check validity
        duration_steps = 1
        if not self.si.isValid(target_out):
            duration_steps = 0

        return duration_steps
