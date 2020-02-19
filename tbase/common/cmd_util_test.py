import random
import unittest

import numpy as np

from tbase.common.cmd_util import set_global_seeds


class TestCmdUtil(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        set_global_seeds(0)

    def test_random(self):
        self.assertEqual(0.8444218515250481, random.random())
        self.assertEqual(0.5488135039273248, np.random.random())


if __name__ == '__main__':
    unittest.main()
