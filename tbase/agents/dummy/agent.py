# -*- coding:utf-8 -*-

import math
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from tbase.agents.base.base_agent import BaseAgent
from tbase.agents.base.explore import explore, simple_explore
from tbase.common.cmd_util import make_env
from tbase.common.logger import logger
from tbase.common.replay_buffer import ReplayBuffer
from tbase.network.polices import get_policy_net


class Agent(BaseAgent):
    def __init__(self, env, args, *other_args):
        # change to random policy
        args.policy_net = "Random"
        super(Agent, self).__init__(env, args, other_args)
        self.policy = get_policy_net(env, args)

        self.num_env = args.num_env
        self.envs = []
        self.states = []
        self.memorys = []
        for i in range(self.num_env):
            env = make_env(args=args)
            state = env.reset()
            self.envs.append(env)
            self.states.append(state)
            self.memorys.append(ReplayBuffer(1e5))
        self.queue = mp.Queue()

    def simple_explore(self):
        t_start = time.time()
        reward_log, portfolios = simple_explore(
            self.envs[0], self.states[0], self.memorys[0],
            self.policy, self.args.explore_size, self.args.print_action)
        used_time = time.time() - t_start
        return np.mean(reward_log), used_time, portfolios

    def explore(self):
        t_start = time.time()
        queue = mp.Queue()
        thread_size = int(math.floor(self.args.explore_size / self.num_env))

        workers = []
        for i in range(self.num_env):
            worker_args = (i, queue, self.envs[i], self.states[i],
                           self.memorys[i], self.policy, thread_size,
                           self.args.print_action)
            workers.append(mp.Process(target=explore, args=worker_args))
        for worker in workers:
            worker.start()

        reward_log = []
        portfolios = []
        for _ in range(self.num_env):
            i, _,  memory, env, state, rewards, portfolio = queue.get()
            self.memorys[i] = memory
            self.envs[i] = env
            self.states[i] = state
            reward_log.extend(rewards)
            portfolios.extend(portfolio)
        used_time = time.time() - t_start

        return np.mean(reward_log), used_time, portfolios

    def learn(self):
        logger.info("learning started")
        i = 0
        current_portfolio = 1.0
        t_start = time.time()
        for i_iter in range(self.args.max_iter_num):
            [avg_reward, e_t, ports] = [None] * 3
            if self.args.num_env == 1:
                avg_reward, e_t, ports = self.simple_explore()
            else:
                avg_reward, e_t, ports = self.explore()
            # NOTE: Don't need update parameters
            for p in ports:
                i += 1
                self.writer.add_scalar('reward/portfolio', p, i)
                current_portfolio = p
                if current_portfolio > self.best_portfolio:
                    self.best_portfolio = current_portfolio
                    logger.info("iter: %d, new best portfolio: %.3f" % (
                        i_iter + 1, self.best_portfolio))
            self.writer.add_scalar('time/explore', e_t, i_iter)

            self.writer.add_scalar('reward/policy',
                                   torch.tensor(avg_reward), i_iter)

            if (i_iter + 1) % self.args.log_interval == 0:
                msg = "total update time: %.1f secs" % (time.time() - t_start)
                msg += ", iter=%d, avg_reward=%.3f" % (i_iter + 1, avg_reward)
                msg += ", current_portfolio: %.3f" % current_portfolio
                logger.info(msg)

        self.writer.close()
        logger.info("Final best portfolio: %.3f" % self.best_portfolio)
        self.save_best_portofolio(self.model_dir)

    def eval(self):
        pass
