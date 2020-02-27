# -*- coding:utf-8 -*-
import math
import os
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from tbase.agents.base.base_agent import BaseAgent
from tbase.agents.base.explore import (buy_and_hold, env_eval, explore,
                                       simple_explore)
from tbase.common.cmd_util import make_env
from tbase.common.logger import logger
from tbase.common.optimizers import get_optimizer_func
from tbase.common.replay_buffer import ReplayBuffer
from tbase.network.polices import get_policy_net
from tbase.network.values import get_value_net


# Actor Critic Agent
class ACAgent(BaseAgent):
    def __init__(self, env, args, *other_args):
        super(ACAgent, self).__init__(env, args, other_args)
        self.num_env = args.num_env
        optimizer_fn = get_optimizer_func(args.opt_fn)()
        # policy net
        self.policy = get_policy_net(env, args)
        self.target_policy = get_policy_net(env, args)
        self.policy_opt = optimizer_fn(
            params=filter(lambda p: p.requires_grad, self.policy.parameters()),
            lr=self.policy.learning_rate)
        # value net
        self.value = get_value_net(env, args)
        self.target_value = get_value_net(env, args)
        self.value_opt = optimizer_fn(
            params=filter(lambda p: p.requires_grad, self.value.parameters()),
            lr=self.args.lr)
        if self.num_env > 1:
            self.queue = mp.Queue()
        self.envs = []
        self.states = []
        self.memorys = []
        for i in range(self.num_env):
            env = make_env(args=args)
            state = env.reset()
            self.envs.append(env)
            self.states.append(state)
            self.memorys.append(ReplayBuffer(1e5))

    def save(self, dir):
        torch.save(
            self.policy.state_dict(),
            '{}/{}.{}.policy.pkl'.format(dir, self.name, self.run_id)
        )
        torch.save(
            self.value.state_dict(),
            '{}/{}.{}.value.pkl'.format(dir, self.name, self.run_id)
        )

    def load(self, dir):
        print(dir)
        if dir is None or not os.path.exists(dir):
            raise ValueError("dir is invalid")
        self.policy.load_state_dict(
            torch.load('{}/{}.{}.policy.pkl'.format(
                dir, self.name, self.run_id))
        )
        self.value.load_state_dict(
            torch.load('{}/{}.{}.value.pkl'.format(
                dir, self.name, self.run_id))
        )

    # 非多进程方式探索与搜集samples
    def simple_explore(self, explore_size=None, sample_size=None):
        t_start = time.time()
        if explore_size is None:
            explore_size = self.args.explore_size
        if sample_size is None:
            sample_size = self.args.sample_size

        reward_log, portfolios = simple_explore(
            self.envs[0], self.states[0], self.memorys[0],
            self.policy, explore_size, self.args.print_action)

        obs, action, rew, obs_next, done = self.memorys[0].sample(
            sample_size)

        used_time = time.time() - t_start

        return obs, action, rew, obs_next, done, \
            np.mean(reward_log), used_time, portfolios

    # 多进程方式探索与搜集samples
    def explore(self, explore_size=None, sample_size=None):
        t_start = time.time()
        if explore_size is None:
            explore_size = self.args.explore_size
        if sample_size is None:
            sample_size = self.args.sample_size
        thread_size = int(math.floor(explore_size / self.num_env))
        thread_sample_size = int(math.floor(sample_size / self.num_env))
        workers = []
        for i in range(self.num_env):
            worker_args = (i, self.queue, self.envs[i], self.states[i],
                           self.memorys[i], self.policy, thread_size,
                           self.args.print_action)
            workers.append(mp.Process(target=explore, args=worker_args))
        for worker in workers:
            worker.start()

        obs, action, rew, obs_next, done = [], [], [], [], []
        reward_log = []
        portfolios = []
        for _ in range(self.num_env):
            i, _,  memory, env, state, rewards, portfolio = self.queue.get()
            self.memorys[i] = memory
            self.envs[i] = env
            self.states[i] = state
            _obs, _action, _rew, _obs_next, _done = self.memorys[i].sample(
                thread_sample_size)
            obs.append(_obs)
            action.append(_action)
            rew.append(_rew)
            obs_next.append(_obs_next)
            done.append(_done)
            reward_log.extend(rewards)
            portfolios.extend(portfolio)
        used_time = time.time() - t_start

        return np.concatenate(tuple(obs), axis=0),\
            np.concatenate(tuple(action), axis=0), \
            np.concatenate(tuple(rew), axis=0), \
            np.concatenate(tuple(obs_next), axis=0), \
            np.concatenate(tuple(done), axis=0), \
            np.mean(reward_log), used_time, portfolios

    def learn(*args):
        raise NotImplementedError

    def warm_up(self):
        logger.info(
            "warmming up: explore %d days in enviroment" % self.args.warm_up)
        if self.num_env > 1:
            self.explore(self.args.warm_up, self.args.sample_size)
        else:
            self.simple_explore(self.args.warm_up, self.args.sample_size)
        logger.info("warm up: finished")

    def eval(self, env, args):
        self.load(self.model_dir)
        _, _, annualized_return, portfolios = env_eval(env,
                                                       self.policy,
                                                       args.print_action)
        bh_annualized_return, bh_portfolios = buy_and_hold(env)
        for i in range(len(portfolios)):
            self.writer.add_scalars('backtesting', {
                self.args.alg: portfolios[i],
                "buy&hold": bh_portfolios[i]}, i)
        excess_return = portfolios[-1] - bh_portfolios[-1]
        logger.info("excess_return: %.3f" % excess_return)
        annual_excess_return = annualized_return - bh_annualized_return
        logger.info("annualized excess_return: %.3f" % annual_excess_return)
