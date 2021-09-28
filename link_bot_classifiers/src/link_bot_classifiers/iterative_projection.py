from typing import Callable
import tensorflow as tf


class BaseProjectOpt:
    def __init__(self, opt):
        self.opt = opt
        self.x_var = None

    def init(self, x_init):
        pass

    def step(self, _, x_var: tf.Variable):
        raise NotImplementedError()


def iterative_projection(initial_value,
                         target,
                         n: int,
                         m: int,
                         step_towards_target: Callable,
                         project_opt: BaseProjectOpt,
                         viz_func: Callable,
                         m_last=None,
                         ):
    x = initial_value
    for i in range(n):
        x, viz_vars = step_towards_target(target, x)
        viz_func(i, x, initial_value, target, viz_vars)

        project_opt.init(x)

        # we might want to spend more iterations satisfying the constraints on the final iteration
        if i == n - 1 and m_last is not None:
            _m = m_last
        else:
            _m = m

        x_var = tf.Variable(x)
        for i in range(_m):
            x, can_terminate, viz_vars = project_opt.step(i, x_var)
            viz_func(i, x, initial_value, target, viz_vars)
            if can_terminate:
                break

    return x
