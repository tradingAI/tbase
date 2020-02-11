# -*- coding:utf-8 -*-
import math
import multiprocessing
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from tbase.common.agent import ACAgent
from tbase.common.cmd_util import common_arg_parser, make_env, set_global_seeds
from tbase.common.logger import logger
from tbase.common.replay_buffer import ReplayBuffer
from tbase.common.torch_utils import clear_memory, device, soft_update
from tbase.network.polices import LSTM_MLP
from tbase.network.values import LSTM_Merge_MLP


# 擦索env,将Transitions存入memory
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
                print("tbase.agents.ddpg.agent explore action:" + str(action))
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


class Agent(ACAgent):
    def __init__(self, env=None, args=None):
        super(Agent, self).__init__(env, args)
        self.args = args
        self.name = self.get_agent_name()
        self.model_dir = self.get_model_dir()
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
        # set seed
        set_global_seeds(args.seed)

    def get_agent_name(self):
        code_str = self.args.codes.replace(",", "_")
        name = "ddpg_" + code_str
        return name

    def get_model_dir(self):
        dir = os.path.join(self.args.model_dir, self.name)
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir

    # 探索与搜集samples
    def explore(self, explore_size, sample_size):
        t_start = time.time()
        queue = multiprocessing.Queue()
        thread_size = int(math.floor(explore_size / self.num_env))
        thread_sample_size = int(math.floor(sample_size / self.num_env))
        workers = []
        for i in range(self.num_env):
            worker_args = (i, queue, self.envs[i], self.states[i],
                           self.memorys[i], self.policy.to('cpu'), thread_size,
                           self.args.print_action)
            workers.append(multiprocessing.Process(target=explore,
                                                   args=worker_args))
        for worker in workers:
            worker.start()

        obs, action, rew, obs_next, done = [], [], [], [], []
        reward_log = []
        portfolios = []
        for _ in workers:
            i, next_idx,  memory, env, state, rewards, portfolio = queue.get()
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

    def update_params(self, _obs, _action, _rew, _obs_next, _done):
        t_start = time.time()
        # --use the date to update the value
        reward = torch.tensor(_rew, device=device, dtype=torch.float)
        reward = reward.reshape(-1, 1)
        done = torch.tensor(~_done, device=device, dtype=torch.float)
        done = done.reshape(-1, 1)
        action = torch.from_numpy(_action).to(device, torch.float)
        action = action.reshape(action.shape[0], -1)
        # obs 只取最后一天数据做为输入
        obs = torch.from_numpy(_obs).permute(1, 0, 2).to(device, torch.float)
        obs_next = torch.from_numpy(_obs_next).permute(1, 0, 2).to(device,
                                                                   torch.float)
        target_act_next = self.target_policy.action(obs_next).detach()

        target_q_next = self.target_value.forward(obs_next, target_act_next)
        target_q = reward + torch.mul(target_q_next, (done * self.args.gamma))
        q = self.value.forward(obs, action)
        # bellman equation
        value_loss = torch.nn.MSELoss()(q, target_q)
        self.value_opt.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value.parameters(),
                                 self.args.max_grad_norm)
        self.value_opt.step()

        # --use the data to update the policy
        # There is no need to cal other agent's action
        action_new, model_out = self.policy.action(obs, with_reg=True)
        # loss_a 表示 value对action的评分负值（-Q值)
        loss_a = torch.mul(-1, torch.mean(self.value.forward(obs, action_new)))
        loss_reg = torch.mean(torch.pow(model_out, 2))
        act_reg = torch.mean(torch.pow(action_new, 2))
        policy_loss = loss_reg + loss_a + act_reg

        # print(action_new.detach().cpu().numpy().tolist())

        self.policy_opt.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(),
                                 self.args.max_grad_norm)
        self.policy_opt.step()

        soft_update(self.target_policy, self.policy, self.args.tau)
        soft_update(self.target_value, self.value, self.args.tau)

        used_time = time.time() - t_start
        return value_loss, policy_loss, loss_reg, act_reg, used_time

    def learn(self):
        logger.info("warmming up: %d" % self.args.warm_up)
        self.explore(self.args.warm_up, self.args.sample_size)
        logger.info("warm up: %d finished" % self.args.warm_up)
        logger.info("learning started")
        i = 0
        current_portfolio = 1.0
        t_start = time.time()
        for i_iter in range(self.args.max_iter_num):
            obs, act, rew, obs_t, done, avg_reward, e_t, ports = self.explore(
                explore_size=self.args.explore_size,
                sample_size=self.args.sample_size)
            for p in ports:
                i += 1
                self.writer.add_scalar('reward/portfolio', p, i)
                current_portfolio = p
            self.writer.add_scalar('time/explore', e_t, i_iter)
            v_loss, p_loss, p_reg, act_reg, u_t = self.update_params(
                obs, act, rew, obs_t, done)
            self.writer.add_scalar('time/update', u_t, i_iter)
            self.writer.add_scalar('loss/value', v_loss, i_iter)
            self.writer.add_scalar('loss/policy', p_loss, i_iter)
            self.writer.add_scalar('reg/action', act_reg, i_iter)
            self.writer.add_scalar('reg/policy', p_reg, i_iter)
            self.writer.add_scalar('reward/policy',
                                   torch.tensor(avg_reward), i_iter)

            if (i_iter + 1) % self.args.log_interval == 0:
                logger.info("time: %.3f, iter=%d, avg_reward=%.3f, last_portfolio: %.3f" % (
                    time.time() - t_start, i_iter,
                    avg_reward, current_portfolio))

            if (i_iter + 1) % self.args.save_model_interval == 0:
                self.save(self.model_dir)
                """clean up gpu memory"""
                clear_memory()


def main():
    args = common_arg_parser()
    if args.debug:
        import logging
        logger.setLevel(logging.DEBUG)
    env = make_env(args=args)
    input_size = env.input_size
    act_size = env.action_space
    policy_net = LSTM_MLP(seq_len=args.look_back_days,
                          input_size=input_size,
                          output_size=act_size)
    target_policy_net = LSTM_MLP(seq_len=args.look_back_days,
                                 input_size=input_size,
                                 output_size=act_size)
    value_net = LSTM_Merge_MLP(seq_len=args.look_back_days,
                               obs_input_size=input_size,
                               act_input_size=act_size)
    target_value_net = LSTM_Merge_MLP(seq_len=args.look_back_days,
                                      obs_input_size=input_size,
                                      act_input_size=act_size)

    def opt_fn(params, lr):
        return torch.optim.RMSprop(params, lr)

    agent = Agent(policy_net, value_net,
                  target_policy_net, target_value_net,
                  env.n, opt_fn, args.tensorboard_dir, args)
    agent.learn()


if __name__ == '__main__':
    main()
