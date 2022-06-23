from typing import Callable

import torch


def iterative_projection(initial_value,
                         target,
                         n: int,
                         m: int,
                         step_towards_target: Callable,
                         project_opt,
                         x_distance: Callable,
                         not_progressing_threshold: float,
                         m_last=None):
    """

    Args:
        initial_value:
        target:
        n: number of outer-loop optimization steps
        m: number of inner-loop projection steps
        step_towards_target:
        project_opt:
        x_distance:
        not_progressing_threshold:
        m_last:

    Returns:

    """
    # in the paper x is $T$, the transformation we apply to the moved points
    x = initial_value

    for i in range(n):
        x_old = x.clone()  # make a copy

        x, viz_vars = step_towards_target(target, x)

        x_param = torch.nn.Parameter(x)
        opt, scheduler = project_opt.make_opt(x_param)

        # we might want to spend more iterations satisfying the constraints on the final step
        if i == n - 1:
            if m_last is not None:
                _m = m_last
            else:
                _m = m * 2  # by default spend twice as many iters on the final step
        else:
            _m = m

        for j in range(_m):
            x, can_terminate, viz_vars = project_opt.project(j, opt, scheduler, x_param)
            if can_terminate:
                break

        # terminate early if we're not progressing
        not_progressing = x_distance(x, x_old) < not_progressing_threshold
        if not_progressing:
            break

    return x, viz_vars
