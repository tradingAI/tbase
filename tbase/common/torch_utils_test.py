import unittest

import numpy as np

from tbase.common.cmd_util import set_global_seeds
from tbase.common.torch_utils import fc, lstm


class TestTorchUtils(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        set_global_seeds(0)

    def sum_params(self, model):
        p_sums = []
        dims = []
        for p in model.parameters():
            dim = list(p.size())
            n = p.cpu().data.numpy()
            p_sums.append(np.sum(n, dtype=np.float))
            dims.append(dim)
        return p_sums, dims

    def test_fc(self):
        layer = fc(10, 2)
        p_sums, dims = self.sum_params(layer)
        self.assertEqual([0.7398650534451008, 0.0],
                         p_sums)
        self.assertEqual([[2, 10], [2]], dims)

    def test_lstm(self):
        rnn = lstm(30, 50, 1, 0)
        p_sums, dims = self.sum_params(rnn)
        self.assertEqual(
            [-4.025678478181362, 11.526431918144226,
             0.8502306267619133, 0.24647170305252075], p_sums)
        self.assertEqual([[200, 30], [200, 50], [200], [200]], dims)


if __name__ == '__main__':
    unittest.main()
