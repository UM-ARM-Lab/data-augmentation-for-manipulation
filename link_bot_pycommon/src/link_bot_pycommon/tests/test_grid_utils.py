from unittest import TestCase

import numpy as np

from link_bot_pycommon.grid_utils import compute_extent_3d, extent_to_env_size, idx_to_point_3d_from_extent, \
    extent_to_env_shape, extent_res_to_origin_point, voxel_grid_to_pc2


class Test(TestCase):
    def test_compute_extent_3d(self):
        actual = compute_extent_3d(rows=200, cols=200, channels=1, resolution=0.01)
        desired = np.array([-1.0, 1.0, -1.0, 1.0, 0.0, 0.01])
        np.testing.assert_allclose(actual, desired)

    def test_compute_extent_3d_2(self):
        actual = compute_extent_3d(rows=200, cols=200, channels=200, resolution=0.01)
        desired = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        np.testing.assert_allclose(actual, desired)

    def test_compute_extent_3d_3(self):
        actual = compute_extent_3d(rows=100, cols=200, channels=1, resolution=0.01)
        desired = np.array([-1.0, 1.0, -0.5, 0.5, 0.0, 0.01])
        np.testing.assert_allclose(actual, desired)

    def test_extent_to_env_size(self):
        extent = [-1, 1, -0.5, 0.5, 0, 0.5]
        env_h_m, env_w_m, env_c_m = extent_to_env_size(extent)
        self.assertEqual(env_h_m, 1)
        self.assertEqual(env_w_m, 2)
        self.assertEqual(env_c_m, 0.5)

    def test_idx_to_point_3d_from_extent(self):
        extent = [0.5, 1.0, 0.0, 1.0, 0.0, 1.0]
        p = idx_to_point_3d_from_extent(row=0, col=0, channel=0, resolution=0.01, extent=extent)
        np.testing.assert_allclose(p, np.array([0.5, 0, 0]))

        p = idx_to_point_3d_from_extent(row=0, col=49, channel=0, resolution=0.01, extent=extent)
        np.testing.assert_allclose(p, np.array([0.99, 0, 0]))

        p = idx_to_point_3d_from_extent(row=99, col=49, channel=0, resolution=0.01, extent=extent)
        np.testing.assert_allclose(p, np.array([0.99, 0.99, 0]))

        p = idx_to_point_3d_from_extent(row=99, col=49, channel=99, resolution=0.01, extent=extent)
        np.testing.assert_allclose(p, np.array([0.99, 0.99, 0.99]))

        p = idx_to_point_3d_from_extent(row=49, col=29, channel=49, resolution=0.01, extent=extent)
        np.testing.assert_allclose(p, np.array([0.79, 0.49, 0.49]))

    def test_extent_to_env_shape(self):
        extent = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        res = 0.01
        h, w, c = extent_to_env_shape(extent, res)
        self.assertEqual(h, 100)
        self.assertEqual(w, 100)
        self.assertEqual(c, 100)

        extent = [0.0, 0.999, 0.0, 1.0, 0.0, 1.0]
        res = 0.01
        h, w, c = extent_to_env_shape(extent, res)
        self.assertEqual(h, 100)
        self.assertEqual(w, 99)
        self.assertEqual(c, 100)

    def test_extent_to_point_of_origin(self):
        extent = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        res = 0.01
        origin_point = extent_res_to_origin_point(extent, res)
        np.testing.assert_allclose(origin_point, np.array([0.0, 0.0, 0.0]))

        extent = [0.3, 1.0, 0.0, 0.5, 0.21, 1.0]
        res = 0.1
        origin_point = extent_res_to_origin_point(extent, res)
        np.testing.assert_allclose(origin_point, np.array([0.30, 0.0, 0.255]))
