def marker_index_generator(idx: int):
    """ Generator for generating unique integers to use in ros visualization msgs MarkerArray messages
    Example:
    ig = marker_index_generator(0)
    next(ig) # 0
    next(ig) # 1
    ig2 = marker_index_generator(1)
    next(ig2) # 100
    next(ig2) # 101

    This way if you want to a plot a marker array which consists of several markers, and you want to control how
    things override things, you can use this to set the idx attribute

    """
    i = 0
    while True:
        yield 100 * idx + i
        i += 1
