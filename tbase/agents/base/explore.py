# -*- coding:utf-8 -*-

import random

import numpy as np
import torch

from tbase.common.eval import annualized_return, max_drawdown, sharpe_ratio
from tbase.common.logger import logger


# 用于多进程擦索env,将Transitions存入memory
def explore(pid, queue, env, state, memory, policy, size, print_actions):
    num_steps = 0
    rewards = []
    portfolios = []
    while num_steps < size:
        state_var = torch.tensor(state).unsqueeze(0).permute(1, 0, 2).to(
            torch.float)
        with torch.no_grad():
            action = policy.select_action(state_var)
            action = action.astype(np.float)
        if print_actions:
            if random.random() < 0.001:
                print("tbase.agents.ddpg.agent action:" + str(action))
        next_state, reward, done, info, _ = env.step(action)
        memory.add(state, action, reward, next_state, done)
        rewards.append(reward)
        num_steps += 1
        if done:
            state = env.reset()
            portfolios.append(info["portfolio_value"])
            continue
        state = next_state
    queue.put([pid, memory._next_idx, memory, env, state, rewards, portfolios])
    return


# 单进程
def simple_explore(env, state, memory, policy, size, print_actions):
    num_steps = 0
    rewards = []
    portfolios = []
    while num_steps < size:
        state_var = torch.tensor(state).unsqueeze(0).permute(1, 0, 2).to(
            torch.float)
        with torch.no_grad():
            action = policy.select_action(state_var)
            action = action.astype(np.float)
        if print_actions:
            if random.random() < 0.001:
                print("tbase.agents.ddpg.agent action:" + str(action))
        next_state, reward, done, info, _ = env.step(action)
        memory.add(state, action, reward, next_state, done)
        rewards.append(reward)
        num_steps += 1
        if done:
            state = env.reset()
            portfolios.append(info["portfolio_value"])
            continue
        state = next_state
    return rewards, portfolios


def env_eval(env, policy, print_actions):
    rewards = []
    daily_returns = []
    portfolios = []
    state = env.reset()
    n_days = 0
    while True:
        state_var = torch.tensor(state).unsqueeze(0).permute(1, 0, 2).to(
            torch.float)
        with torch.no_grad():
            action = policy.action(state_var)
            action = action.detach().cpu()[0].numpy().astype(np.float)
        if print_actions:
            print("tbase.agents.ddpg.agent action:" + str(action))
        next_state, reward, done, info, _ = env.step(action)
        n_days += 1
        rewards.append(reward)
        daily_returns.append(info["daily_pnl"] / env.investment)
        portfolios.append(info["portfolio_value"])
        if done:
            state = env.reset()
            break
        state = next_state
    mdd = max_drawdown(portfolios)
    sharpe_r = sharpe_ratio(daily_returns)
    annualized_return_ = annualized_return(portfolios[-1], n_days)
    logger.info("=" * 38 + "eval" + "=" * 38)
    logger.info("portfolio: %.3f" % portfolios[-1])
    logger.info("max_drawdown: %.3f" % mdd)
    logger.info("sharpe_ratio: %.3f" % sharpe_r)
    logger.info("annualized_return: %.3f" % annualized_return_)
    return mdd, sharpe_r, annualized_return_, portfolios


def buy_and_hold(env):
    """
    在回测第一个交易日均匀分仓买入，并持有到回策结束，用于基线策略
    """
    rewards = []
    daily_returns = []
    portfolios = [1.0]
    env.reset()
    action = env.get_buy_close_action(env.current_date)

    n_days = 0
    while True:
        if n_days < 1:
            _, reward, done, info, _ = env.step(action)
        _, reward, done, info, _ = env.step(action, only_update=True)
        n_days += 1
        rewards.append(reward)
        daily_returns.append(info["daily_pnl"] / env.investment)
        portfolios.append(info["portfolio_value"])
        if done:
            break
    mdd = max_drawdown(portfolios)
    sharpe_r = sharpe_ratio(daily_returns)
    annualized_return_ = annualized_return(portfolios[-1], n_days)
    logger.info("=" * 34 + "buy_and_hold" + "=" * 34)
    logger.info("portfolio: %.3f" % portfolios[-1])
    logger.info("max_drawdown: %.3f" % mdd)
    logger.info("sharpe_ratio: %.3f" % sharpe_r)
    logger.info("annualized_return: %.3f" % annualized_return_)
    return annualized_return_, portfolios
