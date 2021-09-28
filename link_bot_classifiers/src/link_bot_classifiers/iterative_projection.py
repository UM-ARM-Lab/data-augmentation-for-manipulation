from typing import Callable


def iterative_projection(initial_value,
                         target,
                         n: int,
                         m: int,
                         step_x_towards_target: Callable,
                         project_opt,
                         viz_func: Callable,
                         m_last=None,
                         ):
    x = initial_value
    for i in range(n):
        x = x + step_x_towards_target(target, x)
        viz_func(i, x, initial_value, target)

        project_opt.init(x)

        # we might want to spend more iterations satisfying the constraints on the final iteration
        if i == n - 1 and m_last is not None:
            _m = m_last
        else:
            _m = m

        for i in range(_m):
            x = project_opt.step(i, x)
            viz_func(i, x, initial_value, target)
