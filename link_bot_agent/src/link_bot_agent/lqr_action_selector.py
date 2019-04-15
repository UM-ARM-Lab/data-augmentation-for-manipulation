import numpy as np

import control
from link_bot_agent import action_selector


class LQRActionSelector(action_selector.ActionSelector):

    def __init__(self, linear_tf_model, max_v):
        super(LQRActionSelector, self).__init__()
        self.linear_tf_model = linear_tf_model
        self.max_v = max_v
        self.state_matrix, self.control_matrix = self.linear_tf_model.get_dynamics_matrices()
        Q = np.eye(self.linear_tf_model.M)
        # apparently if this R is too small things explode???
        R = np.eye(self.linear_tf_model.L) * 1e-3
        self.K, S, E = control.lqr(self.state_matrix, self.control_matrix, Q, R)

    def act(self, sdf, o_d, o_k, o_d_goal, verbose=False):
        """ return the action which gives the lowest cost for the predicted next state """
        u = np.dot(-self.K, o_d - o_d_goal)
        o_next = self.linear_tf_model.simple_predict(o_d, u.reshape(2, 1))
        return u.reshape(-1, 1, 2), o_next

    def just_d_act(self, o_d, o_d_goal):
        """ return the action which gives the lowest cost for the predicted next state """
        u = np.dot(-self.K, o_d - o_d_goal)
        o_next = self.linear_tf_model.simple_predict(o_d, u.reshape(2, 1))
        return u.reshape(-1, 1, 2), o_next
