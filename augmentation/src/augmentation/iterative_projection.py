from typing import Callable

import tensorflow as tf

from augmentation.aug_opt_utils import debug_aug


class BaseProjectOpt:
    def __init__(self):
        self.opt = None
        self.x_var = None

    def make_opt(self):
        return tf.optimizers.Adam()

    def project(self, i: int, opt, x_var: tf.Variable):
        raise NotImplementedError()


def iterative_projection(initial_value,
                         target,
                         n: int,
                         m: int,
                         step_towards_target: Callable,
                         project_opt: BaseProjectOpt,
                         viz_func: Callable,
                         x_distance: Callable,
                         not_progressing_threshold: float,
                         m_last=None,
                         ):
    # in the paper x is $T$, the transformation we apply to the moved points
    x = initial_value

    if debug_aug():
        viz_func(None, x, initial_value, target, None)
        input("post init. press enter")

    for i in range(n):
        x_old = tf.identity(x)  # make a copy

        x, viz_vars = step_towards_target(target, x)
        if debug_aug():
            for _ in range(3):
                viz_func(i, x, initial_value, target, viz_vars)
            input("post step. press enter")

        opt = project_opt.make_opt()

        # we might want to spend more iterations satisfying the constraints on the final step
        if i == n - 1:
            if m_last is not None:
                _m = m_last
            else:
                _m = m * 2  # by default spend twice as many iters on the final step
        else:
            _m = m

        x_var = tf.Variable(x)
        for j in range(_m):
            x, can_terminate, viz_vars = project_opt.project(j, opt, x_var)
            if debug_aug():
                for _ in range(3):
                    viz_func(i, x, initial_value, target, viz_vars)
            if can_terminate:
                break
        if debug_aug():
            input("post project. press enter")

        # terminate early if we're not progressing
        not_progressing = x_distance(x, x_old) < not_progressing_threshold
        if not_progressing:
            break

    return x, viz_vars
