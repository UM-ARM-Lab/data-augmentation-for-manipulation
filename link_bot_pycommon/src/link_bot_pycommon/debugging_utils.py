SHOW_ALL = False


def debug_viz_batch_indices(batch_size):
    if SHOW_ALL:
        return range(batch_size)
    else:
        return [3]
