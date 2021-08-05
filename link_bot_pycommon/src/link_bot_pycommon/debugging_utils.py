SHOW_ALL = True
DEBUG_VIZ_B = 0


def debug_viz_batch_indices(batch_size):
    if SHOW_ALL:
        return range(batch_size)
    else:
        return [DEBUG_VIZ_B]
