# -*- coding:utf-8 -*-

import numpy as np


def max_drawdown(portfolios):
    """
    最大回撤率
    args:
        portfolios: 净值
    """
    if len(portfolios) == 0:
        return 0
    # 结束位置
    i = np.argmax((np.maximum.accumulate(portfolios) - portfolios) /
                  np.maximum.accumulate(portfolios))
    if i == 0:
        return 0
    j = np.argmax(portfolios[:i])  # 开始位置

    mdd = (portfolios[i] - portfolios[j]) / (portfolios[j])
    return round(mdd, 3)


def annualized_return(portfolio, n_days):
    """
    策略年化收益率
    """
    r = (portfolio - 1.0) * 250 / n_days
    return round(r, 3)


def sharpe_ratio(returns, risk_free=0.000114563492,  periods=252):
    """
    夏普比率
    args:
        returns: 每日收益率, list
        risk_free: 无风险收益率(2020-02-24 国债利率:0.02887), 转化为日化无风险收益率
            用 0.02889 / 252 =  0.00011456349206349206
            NOTE(wen): 这里没有用复利的方法来计算日化无风险收益率
        periods: 考察周期(默认一年取252天)
    """
    if len(returns) == 0:
        return None
    r = np.sqrt(periods) * (np.mean(returns) - risk_free) / np.std(returns)
    return round(r, 3)
