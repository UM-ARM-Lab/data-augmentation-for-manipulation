import matplotlib.pyplot as plt
import numpy as np
import torch

def torch_sdf_3d(x, res):
    """

    Args:
        x: [b, m, n] binary occupancy grid, consisting of 0s and 1s, but of floating point type
        res: [b] meters per cell

    Returns: [b, m, n] approximate signed distance in meters

    """
    b, m, n = x.shape
    sdf = x.clone()
    pool = torch.nn.MaxPool3d(kernel_size=3, stride=1, padding=1)
    for i in range(max(m, n)):
        p = pool(sdf)
        maxp = p.max()
        sdf = sdf + ((maxp + 1) / maxp * p) * (sdf == 0)

    return sdf * res


def torch_sdf_2d(x, res):
    """

    Args:
        x: [b, m, n] binary occupancy grid, consisting of 0s and 1s, but of floating point type
        res: [b] meters per cell

    Returns: [b, m, n] approximate signed distance in meters

    """
    b, m, n = x.shape
    sdf = x.clone()
    pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    iters = max(m, n)
    for i in range(iters):
        from time import perf_counter
        t0 = perf_counter()
        p = pool(sdf)
        maxp = p.max()
        sdf = sdf + ((maxp + 1) / maxp * p) * (sdf == 0)
        print(perf_counter() - t0)

    return sdf * res


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=250)
    b = 64
    m = n = 128
    x = torch.zeros([b, m, n])
    x[0, 0, 0] = 1
    x[1, 33, 25] = 1
    x[2, 53, 15] = 1
    x[3, 13, 85] = 1
    x.cuda()

    from time import perf_counter
    t0 = perf_counter()
    sdf = torch_sdf_2d(x, res=0.01)
    print(f"{perf_counter() - t0:.6f}s")

    for i in [0,1,2,3]:
        plt.figure()
        plt.imshow(sdf[i])
        plt.title(f"{i=}")
        plt.show()


if __name__ == '__main__':
    main()
