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
        expected = [1.5409960746765137, -0.293428897857666]
        self.assertEqual(expected, list(actual.astype(np.float)))
        # action 2
        actual = policy.select_action([])
        expected = [-2.1787893772125244, 0.5684312582015991]
        self.assertEqual(expected, list(actual.astype(np.float)))


if __name__ == '__main__':
    unittest.main()
