import torch

INF = 1e6


def edt_1d(x: torch.Tensor):
    """
    Args:
        x: [n] vector

    Returns:
        same shape as x, the euclidean distance to where x is 0
    """
    d = torch.zeros_like(x)
    n = x.shape[0]
    k = 0
    v = torch.zeros_like(x).long()
    z = torch.zeros_like(x)
    f = 1 - x * INF
    z[0] = -INF
    z[1] = INF
    for q in range(1, n):
        while True:
            s = ((f[q] - q ** 2) - (f[v[k]] + v[k] ** 2)) / (2 * q - 2 * v[k])
            if s > z[k]:
                k += 1
                v[k] = q
                z[k] = s
                z[k + 1] = INF
                break
            else:
                k -= 1
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        d[q] = (q - v[k]) ** 2  # + f[v[k]]

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(d)
    # plt.pause(1)
    return d


def generate_sdf(image):
    a = generate_udf(image)
    b = generate_udf(1 - image)
    return a - b


def generate_udf(image):
    result = (1 - image) * INF

    height, width = result.shape
    capacity = max(width, height)
    i = torch.zeros(result.shape, dtype=torch.long)
    j = torch.zeros(result.shape, dtype=torch.long)
    d = torch.zeros([capacity])
    z = torch.zeros([capacity + 1])
    v = torch.zeros([capacity], dtype=torch.long)

    for x in range(width):
        f = result[:, x]
        edt(f, d, z, v, j[:, x], height)
        result[:, x] = d[:height]
    for y in range(height):
        f = result[y, :]
        edt(f, d, z, v, i[y, :], width)
        result[y, :] = d[:width]

    return torch.sqrt(result)


def edt(f, d, z, v, i, n):
    """

    Args:
        f: source data (returns the Y of the parabola vertex at X)
        d: destination data (final distance values are written here)
        z: temporary used to store X coords of parabola intersections
        v: temporary used to store X coords of parabola vertices
        i: resulting X coords of parabola vertices
        n: number of pixels in "f" to process

    Returns:
        None
    """

    # Always add the first pixel to the enveloping set since it is
    # obviously lower than all parabolas processed so far.
    k: int = 0
    v[0] = 0
    z[0] = -INF
    z[1] = +INF

    for q in range(1, n):

        # If the new parabola is lower than the right-most parabola in
        # the envelope, remove it from the envelope. To make this
        # determination, find the X coordinate of the intersection (s)
        # between the parabolas with vertices at (q,f[q]) and (p,f[p]).
        p = v[k]
        s = ((f[q] + q * q) - (f[p] + p * p)) / (2.0 * q - 2.0 * p)
        while s <= z[k]:
            k = k - 1
            p = v[k]
            s = ((f[q] + q * q) - (f[p] + p * p)) / (2.0 * q - 2.0 * p)

        # Add the new parabola to the envelope.
        k = k + 1
        v[k] = q
        z[k] = s
        z[k + 1] = +INF

    # Go back through the parabolas in the envelope and evaluate them
    # in order to populate the distance values at each X coordinate.
    k = 0
    for q in range(n):
        while z[k + 1] < float(q):
            k = k + 1
        dx = q - v[k]
        d[q] = dx * dx + f[v[k]]
        i[q] = v[k]
