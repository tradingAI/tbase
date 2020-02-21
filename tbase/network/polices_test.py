import unittest

import numpy as np

from tbase.common.cmd_util import set_global_seeds
from tbase.network.polices import RandomPolicy


class TestPolices(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        set_global_seeds(0)

    def test_random_policy(self):
        policy = RandomPolicy(2)
        # action 1
        actual = policy.select_action([])
        expected = [1.0, -0.2534131770209437]
        self.assertEqual(expected, list(actual.astype(np.float)))
        # action 2
        actual = policy.select_action([])
        expected = [-1.0, 0.8324962832376306]
        self.assertEqual(expected, list(actual.astype(np.float)))


if __name__ == '__main__':
    unittest.main()
