#!/usr/bin/env python
import unittest

import numpy as np

from moonshine.gpu_config import limit_gpu_mem
from moonshine.raster_3d_tf import raster_3d

limit_gpu_mem(0.1)


class Test(unittest.TestCase):

    def test_raster_3d(self):
        state = np.array([[0, 0, 0, 0, 0.02, 0, -0.01, 0.01, 0.01]], dtype=np.float32)
        res = [0.01]
        h = 5
        w = 5
        c = 5
        origin = np.array([[h // 2, w // 2, c // 2]], dtype=np.float32)
        k = 100000
        batch_size = state.shape[0]

        image = raster_3d(state, res, origin, h, w, c, k, batch_size)

        self.assertAlmostEqual(image[0, 0, 0, 0, 0].numpy(), 0)
        self.assertAlmostEqual(image[0, 1, 1, 1, 0].numpy(), 0)
        self.assertAlmostEqual(image[0, 2, 2, 2, 0].numpy(), 1)
        self.assertAlmostEqual(image[0, 4, 2, 2, 1].numpy(), 1)
        self.assertAlmostEqual(image[0, 3, 1, 3, 2].numpy(), 1)
