#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

import os
import json

import numpy as np
import control
import tensorflow as tf
from colorama import Fore
from link_bot_notebooks import base_model
from link_bot_notebooks import toy_problem_optimization_common as tpo
from tensorflow.python import debug as tf_debug
import matplotlib.pyplot as plt


@tf.custom_gradient
def sdf_func(sdf, full_sdf_gradient, resolution, sdf_origin_coordinate, sdf_coordinates, P, Q):
    integer_coordinates = tf.cast(tf.divide(sdf_coordinates, resolution), dtype=tf.int32)
    integer_coordinates = tf.reshape(integer_coordinates, [-1, P])
    integer_coordinates = integer_coordinates + sdf_origin_coordinate
    # blindly assume the point is within our grid

    # https://github.com/tensorflow/tensorflow/pull/15857
    # "on CPU an error will be returned and on GPU 0 value will be filled to the expected positions of the output."
    # TODO: make this handle out of bounds correctly. I think correctly for us means return large number for SDF
    # and a gradient towards the origin
    sdf_value = tf.gather_nd(sdf, integer_coordinates, name='index_sdf')
    sdf_value = tf.reshape(sdf_value, (sdf_coordinates.shape[0], sdf_coordinates.shape[1], Q))

    # noinspection PyUnusedLocal
    def __sdf_gradient_func(dy):
        sdf_gradient = tf.gather_nd(full_sdf_gradient, integer_coordinates, name='index_sdf_gradient')
        sdf_gradient = tf.reshape(sdf_gradient, (sdf_coordinates.shape[0], sdf_coordinates.shape[1], P))
        return None, None, None, None, sdf_gradient, None, None

    return sdf_value, __sdf_gradient_func


class LinearConstraintModel(base_model.BaseModel):

    def __init__(self, args, numpy_sdf, numpy_sdf_gradient, numpy_sdf_resolution, batch_size, N, M, L, P, Q, dt,
                 n_steps, seed=0):
        base_model.BaseModel.__init__(self, N, M, L, P)

        self.seed = seed
        np.random.seed(self.seed)
        tf.random.set_random_seed(self.seed)

        self.batch_size = batch_size
        self.args = args
        self.N = N
        self.M = M
        self.L = L
        self.P = P
        self.Q = Q
        self.beta = 1e-8
        self.n_steps = n_steps
        self.dt = dt
        self.sdf_rows, self.sdf_cols = numpy_sdf.shape
        self.sdf_origin_coordinate = np.array([self.sdf_rows / 2, self.sdf_cols / 2], dtype=np.int32)

        self.s = tf.placeholder(tf.float32, shape=(batch_size, self.n_steps + 1, N), name="s")
        self.u = tf.placeholder(tf.float32, shape=(batch_size, self.n_steps, L), name="u")
        self.s_goal = tf.placeholder(tf.float32, shape=(1, N), name="s_goal")
        self.c_label = tf.placeholder(tf.float32, shape=(batch_size, self.n_steps + 1), name="c")
        self.k_label = tf.placeholder(tf.float32, shape=(batch_size, self.n_steps + 1, Q), name="k")

        R_d_init = np.random.randn(N, M).astype(np.float32) * 1e-6
        R_d_init[0, 0] = 1
        R_d_init[1, 1] = 1
        R_k_init = np.random.randn(N, P).astype(np.float32) * 1e-6
        R_k_init[4, 0] = 1
        R_k_init[5, 1] = 1
        A_d_init = np.random.randn(M, M).astype(np.float32) * 1e-6
        B_d_init = np.random.randn(M, L).astype(np.float32) * 1e-6
        A_k_init = np.random.randn(M, M).astype(np.float32) * 1e-6
        B_k_init = np.random.randn(M, L).astype(np.float32) * 1e-6
        np.fill_diagonal(B_d_init, 1)
        np.fill_diagonal(B_k_init, 1)

        # Fake linear data
        # A_d_init = np.array([[0.1, 0.2], [0.3, 0.4]]).astype(np.float32)
        # B_d_init = np.array([[2, 1], [0, 3]]).astype(np.float32)
        # A_k_init = np.zeros((M, M)).astype(np.float32) * 1e-6
        # B_k_init = np.zeros((M, L)).astype(np.float32) * 1e-6

        self.R_d = tf.get_variable("R_d", initializer=R_d_init)
        self.A_d = tf.get_variable("A_d", initializer=A_d_init)
        self.B_d = tf.get_variable("B_d", initializer=B_d_init)

        self.R_k = tf.get_variable("R_k", initializer=R_k_init)
        self.A_k = tf.get_variable("A_k", initializer=A_k_init)
        self.B_k = tf.get_variable("B_k", initializer=B_k_init)

        # self.threshold_k = tf.get_variable("threshold_k", initializer=1.0)
        self.threshold_k = tf.get_variable("threshold_k", initializer=0.15, trainable=False)

        # we force D to be identity because it's tricky to constrain it to be positive semi-definite
        self.D = tf.get_variable("D", initializer=np.eye(self.M, dtype=np.float32), trainable=False)

        self.hat_o_d = tf.einsum('bsn,nm->bsm', self.s, self.R_d, name='hat_o_d')
        self.hat_o_k = tf.einsum('bsn,nm->bsm', self.s, self.R_k, name='hat_o_k')
        self.o_d_goal = tf.matmul(self.s_goal, self.R_d, name='og')

        hat_o_d_next = [self.hat_o_d[:, 0, :]]
        hat_o_k_next = [self.hat_o_k[:, 0, :]]

        for i in range(1, self.n_steps + 1):
            Adod = tf.einsum('mp,bp->bm', self.dt * self.A_d, hat_o_d_next[i - 1], name='A_d_o_d')
            Bdu = tf.einsum('ml,bl->bm', self.dt * self.B_d, self.u[:, i - 1], name='B_d_u')
            hat_o_d_next.append(hat_o_d_next[i - 1] + Adod + Bdu)
            Akok = tf.einsum('mp,bp->bm', self.dt * self.A_k, hat_o_k_next[i - 1], name='A_k_o_k')
            Bku = tf.einsum('ml,bl->bm', self.dt * self.B_k, self.u[:, i - 1], name='B_k_u')
            hat_o_k_next.append(hat_o_k_next[i - 1] + Akok + Bku)

        self.hat_o_d_next = tf.transpose(tf.stack(hat_o_d_next), [1, 0, 2], name='hat_o_d_next')
        self.hat_o_k_next = tf.transpose(tf.stack(hat_o_k_next), [1, 0, 2], name='hat_o_k_next')

        self.d_to_goal = self.o_d_goal - self.hat_o_d_next
        self.hat_c = tf.einsum('bst,tp,bsp->bs', self.d_to_goal, self.D, self.d_to_goal)
        self.sdfs = sdf_func(numpy_sdf, numpy_sdf_gradient, numpy_sdf_resolution, self.sdf_origin_coordinate,
                             self.hat_o_k
                             , self.P, self.Q)
        self.hat_k = self.sdfs - self.threshold_k
        self.hat_k_violated = tf.cast(self.sdfs < self.threshold_k, dtype=tf.int32)
        self.k_label_binary = tf.cast((self.k_label+1)/2, dtype=tf.int32)
        _, self.constraint_prediction_accuracy = tf.metrics.accuracy(self.hat_k_violated, self.k_label_binary)

        # NOTE: we use a mask to set the state prediction loss to 0 when the constraint is violated?
        # this way we don't penalize our model for failing to predict the dynamics in collision
        self.constraint_label_mask = tf.squeeze(self.k_label_binary)

        with tf.name_scope("train"):
            # sum of squared errors in latent space at each time step
            with tf.name_scope("latent_dynamics_d"):
                self.state_prediction_error_in_d = tf.reduce_sum(tf.pow(self.hat_o_d - self.hat_o_d_next, 2), axis=2)
                # self.state_prediction_error_in_d = self.state_prediction_error_in_d * self.constraint_label_mask
                self.state_prediction_loss_in_d = tf.reduce_mean(self.state_prediction_error_in_d,
                                                                 name='state_prediction_loss_in_d')
                self.cost_prediction_error = self.hat_c - self.c_label
                # self.cost_prediction_error = self.cost_prediction_error * self.constraint_label_mask
                self.cost_prediction_loss = tf.reduce_mean(self.cost_prediction_error, name='cost_prediction_loss')

            with tf.name_scope("latent_constraints_k"):
                self.state_prediction_error_in_k = tf.reduce_sum(tf.pow(self.hat_o_k - self.hat_o_k_next, 2), axis=2)
                # self.state_prediction_error_in_k = self.state_prediction_error_in_k * self.constraint_label_mask
                self.state_prediction_loss_in_k = tf.reduce_mean(self.state_prediction_error_in_k,
                                                                 name='state_prediction_loss_in_k')
                # if the hat_k > 0, then we are predicting no collision
                # if k_label is = 1, the true label is collision
                # multiplying these two positive numbers gives a high loss, because our prediction is wrong.
                # if hat_k>0 and k_label=1 we get high loss,
                # if hat_k>0 and k_label=-1 we get low loss,
                # if hat_k<0 and k_label=1 we get high loss,
                # if hat_k<0 and k_label=-1 we get low loss,
                self.constraint_prediction_loss = tf.reduce_mean(self.hat_k * self.k_label,
                                                                 name="constraint_prediction_loss")

            self.flat_weights = tf.concat(
                (tf.reshape(self.R_d, [-1]), tf.reshape(self.A_d, [-1]), tf.reshape(self.B_d, [-1])), axis=0)
            self.regularization = tf.nn.l2_loss(self.flat_weights) * self.beta

            # self.loss = tf.add_n(
            #     [self.state_prediction_loss_in_d, self.state_prediction_loss_in_k, self.cost_prediction_loss,
            #      self.constraint_prediction_loss, self.regularization])
            self.loss = tf.add_n([self.constraint_prediction_loss])

            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            self.opt = tf.train.AdamOptimizer(learning_rate=0.002).minimize(self.loss, global_step=self.global_step)

            trainable_vars = tf.trainable_variables()
            for var in trainable_vars:
                name = var.name.replace(":", "_")
                grads = tf.gradients(self.loss, var, name='dLoss_d{}'.format(name))
                for grad in grads:
                    if grad is not None:
                        tf.summary.histogram(name + "/gradient", grad)
                    else:
                        print("Warning... there is no gradient of the loss with respect to {}".format(var.name))

            tf.summary.scalar("state_prediction_loss_in_d", self.state_prediction_loss_in_d)
            tf.summary.scalar("cost_prediction_loss", self.cost_prediction_loss)
            tf.summary.scalar("regularization_loss", self.regularization)
            tf.summary.scalar("loss", self.loss)

            self.summaries = tf.summary.merge_all()
            self.sess = tf.Session()
            if args['debug']:
                self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
            self.saver = tf.train.Saver(max_to_keep=None)

    def train(self, train_x, goal, epochs, log_path):
        interrupted = False

        writer = None
        loss = None
        full_log_path = None
        if self.args['log'] is not None:
            full_log_path = os.path.join("log_data", log_path)

            tpo.make_log_dir(full_log_path)

            metadata_path = os.path.join(full_log_path, "metadata.json")
            metadata_file = open(metadata_path, 'w')
            metadata = {
                'tf_version': str(tf.__version__),
                'log path': full_log_path,
                'checkpoint': self.args['checkpoint'],
                'N': self.N,
                'M': self.M,
                'L': self.L,
                'beta': self.beta,
                'dt': self.dt,
                'commandline': self.args['commandline'],
            }
            metadata_file.write(json.dumps(metadata, indent=2))

            writer = tf.summary.FileWriter(full_log_path)
            writer.add_graph(self.sess.graph)

        try:
            s, u, c, k = self.compute_cost_label_and_seperate_data(train_x, goal)
            feed_dict = {self.s: s,
                         self.u: u,
                         self.s_goal: goal,
                         self.c_label: c,
                         self.k_label: k}

            ops = [self.global_step, self.summaries, self.loss, self.opt]
            for i in range(epochs):
                step, summary, loss, _ = self.sess.run(ops, feed_dict=feed_dict)

                if 'save_period' in self.args and (step % self.args['save_period'] == 0 or step == 1):
                    if self.args['log'] is not None:
                        writer.add_summary(summary, step)
                        self.save(full_log_path, loss=loss)

                if 'print_period' in self.args and (step % self.args['print_period'] == 0 or step == 1):
                    print(step, loss)

        except KeyboardInterrupt:
            print("stop!!!")
            interrupted = True
            pass
        finally:
            ops = [self.R_d, self.A_d, self.B_d, self.D]
            A, B, C, D = self.sess.run(ops, feed_dict={})
            if self.args['verbose']:
                print("Loss: {}".format(loss))
                print("A:\n{}".format(A))
                print("B:\n{}".format(B))
                print("C:\n{}".format(C))
                print("D:\n{}".format(D))

        return interrupted

    def evaluate(self, eval_x, goal, display=True):
        s, u, c, k = self.compute_cost_label_and_seperate_data(eval_x, goal)
        feed_dict = {self.s: s,
                     self.u: u,
                     self.s_goal: goal,
                     self.c_label: c,
                     self.k_label: k}
        ops = [self.R_d, self.A_d, self.B_d, self.D, self.R_k, self.A_k, self.B_k, self.threshold_k,
               self.state_prediction_loss_in_d, self.state_prediction_loss_in_k, self.cost_prediction_loss,
               self.constraint_prediction_loss, self.regularization, self.loss, self.constraint_prediction_accuracy]
        R_d, A_d, B_d, D, R_k, A_k, B_k, threshold_k, spd_loss, spk_loss, c_loss, k_loss, reg, loss, k_accuracy = \
            self.sess.run(ops, feed_dict=feed_dict)

        print(self.sess.run([(self.k_label+1)/2, self.hat_k_violated], feed_dict=feed_dict))

        if display:
            print("State Prediction Loss in d: {}".format(spd_loss))
            print("State Prediction Loss in k: {}".format(spk_loss))
            print("Cost Loss: {}".format(c_loss))
            print("Constraint Loss: {}".format(k_loss))
            print("Regularization: {}".format(reg))
            print("Overall Loss: {}".format(loss))
            print("R_d:\n{}".format(R_d))
            print("A_d:\n{}".format(A_d))
            print("B_d:\n{}".format(B_d))
            print("D:\n{}".format(D))
            print("R_k:\n{}".format(R_k))
            print("A_k:\n{}".format(A_k))
            print("B_k:\n{}".format(B_k))
            print("threashold_k:\n{}".format(threshold_k))
            print("constraint prediction accuracy:\n{}".format(k_accuracy))
            controllable = self.is_controllable()
            if controllable:
                controllable_string = Fore.GREEN + "True" + Fore.RESET
            else:
                controllable_string = Fore.RED + "False" + Fore.RESET
            print("Controllable?: " + controllable_string)

            # visualize a few sample predictions from the testing data
            self.sess.run([self.hat_o_d_next], feed_dict=feed_dict)

        return R_d, A_d, B_d, D, R_k, A_k, B_k, c_loss, spd_loss, spk_loss, c_loss, k_loss

    def compute_cost_label_and_seperate_data(self, x, goal):
        """ x is 3d.
            first axis is the trajectory.
            second axis is the time step
            third axis is the [state|action] data
        """
        # this ordering is prescribed by the cord in agent.py
        s = x[:, :, 2:2 + self.N]
        k = x[:, :, 1].reshape(x.shape[0], x.shape[1], self.Q)
        u = x[:, :-1, -self.L:]
        # NOTE: Here we compute the label for cost/reward and constraints
        c = np.sum((s[:, :, [0, 1]] - goal[0, [0, 1]]) ** 2, axis=2)
        return s, u, c, k

    def setup(self):
        if self.args['checkpoint']:
            self.load()
        else:
            self.init()

    def init(self):
        self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    def reduce(self, s):
        ss = np.ndarray((self.batch_size, self.n_steps + 1, self.N))
        ss[0, 0] = s
        feed_dict = {self.s: ss}
        ops = [self.hat_o_d]
        hat_o = self.sess.run(ops, feed_dict=feed_dict)[0]
        return hat_o[0, 0].reshape(self.M, 1)

    def predict(self, o, u):
        """
        :param o: 1xM or Mx1
        :param u: batch_size x n_steps x L
        """
        hat_o = np.ndarray((self.batch_size, self.n_steps + 1, self.M))
        hat_o[0, 0] = np.squeeze(o)
        feed_dict = {self.hat_o_d: hat_o, self.u: u}
        ops = [self.hat_o_d_next]
        hat_o_next = self.sess.run(ops, feed_dict=feed_dict)[0]
        return hat_o_next

    def simple_predict(self, o, u):
        R_d, A_d, B_d, D, R_k, A_k, B_k = self.get_matrices()
        o_next = o + self.dt * np.dot(B_d, o) + self.dt * np.dot(B_d, u)
        return o_next

    def predict_cost(self, o, u, g):
        hat_o = np.ndarray((self.batch_size, self.n_steps + 1, self.M))
        hat_o[0, 0] = np.squeeze(o)
        feed_dict = {self.hat_o_d: hat_o, self.u: u, self.s_goal: g}
        ops = [self.hat_c]
        hat_c_next = self.sess.run(ops, feed_dict=feed_dict)[0]
        return hat_c_next

    def hat_constraint(self, s):
        feed_dict = {self.s: s}
        ops = [self.hat_o_k]
        hat_constraint = self.sess.run(ops, feed_dict=feed_dict)[0]
        return hat_constraint

    def predict_from_s(self, s, u):
        feed_dict = {self.s: s, self.u: u}
        ops = [self.hat_o_d_next]
        hat_o_next = self.sess.run(ops, feed_dict=feed_dict)[0]
        return hat_o_next

    def cost_of_s(self, s, g):
        return self.cost(self.reduce(s), g)

    def cost(self, o, g):
        hat_o_next = np.zeros((self.batch_size, self.n_steps + 1, self.M))
        for i in range(self.batch_size):
            for j in range(self.n_steps + 1):
                hat_o_next[i, j] = np.squeeze(o)
        feed_dict = {self.hat_o_d_next: hat_o_next, self.s_goal: g}
        ops = [self.hat_c]
        hat_c = self.sess.run(ops, feed_dict=feed_dict)[0]
        hat_c = np.expand_dims(hat_c, axis=0)
        return hat_c

    def save(self, log_path, log=True, loss=None):
        global_step = self.sess.run(self.global_step)
        if log:
            if loss:
                print(Fore.CYAN + "Saving ckpt {} at step {:d} with loss {}".format(log_path, global_step,
                                                                                    loss) + Fore.RESET)
            else:
                print(Fore.CYAN + "Saving ckpt {} at step {:d}".format(log_path, global_step) + Fore.RESET)
        self.saver.save(self.sess, os.path.join(log_path, "nn.ckpt"), global_step=self.global_step)

    def load(self):
        self.saver.restore(self.sess, self.args['checkpoint'])
        global_step = self.sess.run(self.global_step)
        print(Fore.CYAN + "Restored ckpt {} at step {:d}".format(self.args['checkpoint'], global_step) + Fore.RESET)

    def is_controllable(self):
        feed_dict = {}
        ops = [self.A_d, self.B_d]
        # Normal people use A and B here but I picked stupid variable names
        state_matrix, control_matrix = self.sess.run(ops, feed_dict=feed_dict)
        controllability_matrix = control.ctrb(state_matrix, control_matrix)
        rank = np.linalg.matrix_rank(controllability_matrix)
        return rank == self.M

    def get_matrices(self):
        feed_dict = {}
        ops = [self.R_d, self.A_d, self.B_d, self.D, self.R_k, self.A_k, self.B_k]
        return self.sess.run(ops, feed_dict=feed_dict)

    def get_R_d(self):
        feed_dict = {}
        ops = [self.R_d]
        R_d = self.sess.run(ops, feed_dict=feed_dict)[0]
        return R_d

    def __str__(self):
        R_d, A_d, B_d, D, R_k, A_k, B_k = self.get_matrices()
        return "R_d:\n" + np.array2string(R_d) + "\n" + \
               "A_d:\n" + np.array2string(A_d) + "\n" + \
               "B_d:\n" + np.array2string(B_d) + "\n" + \
               "D:\n" + np.array2string(D) + "\n"
