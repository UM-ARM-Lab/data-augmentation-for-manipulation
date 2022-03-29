import unittest

import numpy as np
import torch

from moonshine.torch_edt import edt_1d, generate_sdf


class TestIndexing(unittest.TestCase):

    def test_edt_1d(self):
        x = torch.tensor([0, 0, 0, 1, 0, 0, 0], dtype=torch.float32)
        expected = torch.tensor([4, 2, 1, 0, 1, 2, 4], dtype=torch.float32)
        out = edt_1d(x)
        np.testing.assert_allclose(out.numpy(), expected.numpy())


if __name__ == '__main__':
    # unittest.main()
    # x = torch.tensor([0, 0, 0, 1, 0, 0, 0], dtype=torch.float32)
    # expected = torch.tensor([9, 4, 1, 0, 1, 4, 9], dtype=torch.float32)
    # out = edt_1d(x)
    # np.testing.assert_allclose(out.numpy(), expected.numpy())
    #
    # x = torch.tensor([0, 0, 0, 1, 1, 0, 0], dtype=torch.float32)
    # expected = torch.tensor([9, 4, 1, 0, 0, 1, 4], dtype=torch.float32)
    # out = edt_1d(x)
    # np.testing.assert_allclose(out.numpy(), expected.numpy())

    image = torch.zeros([128, 128])
    image[2:4, 5:7] = 1
    image[20:40, 45:70] = 1
    from time import perf_counter
    t0 = perf_counter()
    sdf = generate_sdf(image)
    print(perf_counter() - t0)
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(image)
    plt.figure()
    plt.imshow(sdf)
    plt.show(block=True)
