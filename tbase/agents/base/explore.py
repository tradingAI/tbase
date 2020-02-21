# -*- coding:utf-8 -*-

import random

import numpy as np
import torch


# 多进程擦索env,将Transitions存入memory
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
