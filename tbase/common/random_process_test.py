import unittest

import numpy as np

from tbase.common.cmd_util import set_global_seeds
from tbase.common.random_process import (GaussianProcess,
                                         OrnsteinUhlenbeckProcess)


class TestRandomProcess(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        set_global_seeds(0)
        self.ou = OrnsteinUhlenbeckProcess(0.1, size=4)
        self.gaussian = GaussianProcess(mu=0., sigma=0.5, size=2)

    def test_all(self):
        # 10 steps
        actual = []
        for i in range(10):
            actual.append(np.sum(self.ou.sample(), dtype=np.float))
        expected = [0.5383840737642085, 0.706746821640589, 0.8966097476458151,
                    1.061738168074121, 1.135465707080145, 0.9566202371789995,
                    1.023059978608266, 1.3755617115854408, 1.0681716102662342,
                    1.2414075348088525]
        self.assertEqual(expected, actual)
        # 1000 steps
        actual = 0
        for i in range(1000):
            actual += np.sum(self.ou.sample(), dtype=np.float)
        expected = -3772.8442412177346
        self.assertEqual(expected, actual)
        # test guassian
        # 10 steps
        actual = []
        for i in range(10):
            actual.append(np.sum(self.gaussian.sample(), dtype=np.float))
        expected = [
            0.07241258089459185, 0.36464358642918343, -0.6695803345099468,
            0.9502863078869512, 0.7670462257436717, -1.052869747466798,
            -0.23612638092134347, -0.3332191841176338, -0.12752003461373218,
            0.09916366243489146]
        self.assertEqual(expected, actual)
        # 1000 steps
        actual = 0
        for i in range(1000):
            actual += np.sum(self.gaussian.sample(), dtype=np.float)
        expected = 4.857618926778793
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
