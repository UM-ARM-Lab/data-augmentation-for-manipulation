import unittest

import tensorflow as tf
import torch

from moonshine.indexing import index_dict_of_batched_tensors_tf
from moonshine.tests import testing_utils
import numpy as np

from moonshine.torch_edt import edt_1d


class TestIndexing(unittest.TestCase):

    def test_edt_1d(self):
        x = torch.tensor([0, 0, 0, 1, 0, 0, 0], dtype=torch.float32)
        expected = torch.tensor([4, 2, 1, 0, 1, 2, 4], dtype=torch.float32)
        out = edt_1d(x)
        np.testing.assert_allclose(out.numpy(), expected.numpy())


if __name__ == '__main__':
    # unittest.main()
    x = torch.tensor([0, 0, 0, 1, 0, 0, 0], dtype=torch.float32)
    expected = torch.tensor([9, 4, 1, 0, 1, 4, 9], dtype=torch.float32)
    out = edt_1d(x)
    np.testing.assert_allclose(out.numpy(), expected.numpy())

    x = torch.tensor([0, 0, 0, 1, 1, 0, 0], dtype=torch.float32)
    expected = torch.tensor([9, 4, 1, 0, 0, 1, 4], dtype=torch.float32)
    out = edt_1d(x)
    np.testing.assert_allclose(out.numpy(), expected.numpy())
