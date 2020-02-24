# -*- coding:utf-8 -*-

import unittest

from tbase.common.eval import annualized_return, max_drawdown, sharpe_ratio


class TestEval(unittest.TestCase):
    def test_max_drawdown(self):
        portfolios = [9.0, 7.0, 7.0, 10.0, 5.0, 9.0]
        self.assertEqual(-0.5, max_drawdown(portfolios))
        self.assertEqual(0, max_drawdown([]))

    def test_annualized_returns(self):
        portfolio = 1.1
        n_days = 125
        self.assertEqual(0.2, annualized_return(portfolio, n_days))

    def test_shape_ratio(self):
        returns = [0.01, 0.02, -0.015, 0.03]
        self.assertEqual(10.57, sharpe_ratio(returns))


if __name__ == '__main__':
    unittest.main()
