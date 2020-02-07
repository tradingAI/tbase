# -*- coding:utf-8 -*-
import math
import multiprocessing

import numpy as np
import torch

from tbase.common.agent import ACAgent
from tbase.common.cmd_util import common_arg_parser, make_env, set_global_seeds
from tbase.common.logger import logger
from tbase.common.replay_buffer import ReplayBuffer


# 擦索env,将Transitions存入memory
def explore(pid, env, states, memory, policy, size):
    num_steps = 0
    # TODO: 验证 list 能否在多进程中更新
    state = states[pid]
    while num_steps < size:
        while True:
            state_var = torch.tensor(state).unsqueeze(0).permute(1, 0, 2).to(
                torch.float)
            with torch.no_grad():
                action = policy.select_action(state_var).numpy()
            action = action.astype(np.float)
            next_state, reward, done, info, _ = env.step(action)
            # TODO: memory add 是否在多线程环境下发生冲突
            memory.add(state, action, reward, next_state, done)
            num_steps += 1

            if done:
                state = env.reset()
                break
            state = next_state
    states[pid] = state


class Agent(ACAgent):
    def __init__(self, policy_net=None, value_net=None, args=None):
        super(Agent, self).__init__(policy_net, value_net, args)
        self.num_env = args.num_env
        self.envs = []
        self.states = []
        for i in range(self.num_env):
            env = make_env(args=args)
            state = env.reset()
            self.envs.append(env)
            self.states.append(state)
        self.memory = ReplayBuffer(10e5)
        if not(args.seed is None):
            set_global_seeds(args.seed)

    # 探索与搜集samples
    def explore(self, explore_size, sample_size):
        thread_size = int(math.floor(explore_size / self.num_env))
        workers = []
        for i in range(self.num_env):
            worker_args = (i, self.env[i], self.policy, thread_size)
            workers.append(multiprocessing.Process(target=explore,
                                                   args=worker_args))
        for worker in workers:
            worker.start()

        batch = self.memory.sample(sample_size)

        return batch

    def learn(self):
        logger.info("learning")
        # TODO


def main():
    policy_net = None
    value_net = None
    args = common_arg_parser()
    agent = Agent(policy_net, value_net, args)
    agent.learn()


if __name__ == '__main__':
    main()
