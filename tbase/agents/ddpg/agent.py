# -*- coding:utf-8 -*-
import math
import multiprocessing
import os

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
def explore(pid, queue, env, state, memory, policy, size):
    num_steps = 0
    rewards = []
    while num_steps < size:
        state_var = torch.tensor(state).unsqueeze(0).permute(1, 0, 2).to(
            torch.float)
        with torch.no_grad():
            action = policy.select_action(state_var)
        action = action.astype(np.float)
        next_state, reward, done, info, _ = env.step(action)
        memory.add(state, action, reward, next_state, done)
        rewards.append(reward)
        num_steps += 1
        if done:
            state = env.reset()
            continue
        state = next_state
    queue.put([pid, memory._next_idx, memory, env, state, rewards])


class Agent(ACAgent):
    def __init__(self, policy_net=None, value_net=None,
                 target_policy_net=None, target_value_net=None,
                 n_codes=1, optimizer_fn=None, args=None):
        super(Agent, self).__init__(
            policy_net, value_net,
            target_policy_net, target_value_net,
            optimizer_fn, args)
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
        if not(args.seed is None):
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
        queue = multiprocessing.Queue()
        thread_size = int(math.floor(explore_size / self.num_env))
        print(thread_size)
        thread_sample_size = int(math.floor(sample_size / self.num_env))
        workers = []
        for i in range(self.num_env):
            worker_args = (i, queue, self.envs[i], self.states[i],
                           self.memorys[i], self.policy, thread_size)
            workers.append(multiprocessing.Process(target=explore,
                                                   args=worker_args))
        for worker in workers:
            worker.start()

        obs, action, rew, obs_next, done = [], [], [], [], []
        reward_log = []
        for _ in workers:
            i, next_idx,  memory, env, state, rewards = queue.get()
            self.memorys[i] = memory
            self.envs[i] = env
            self.states[i] = state
            print(i, next_idx, self.memorys[i]._next_idx)
            _obs, _action, _rew, _obs_next, _done = self.memorys[i].sample(
                thread_sample_size)
            obs.append(_obs)
            action.append(_action)
            rew.append(_rew)
            obs_next.append(_obs_next)
            done.append(_done)
            reward_log.extend(rewards)
        print("avg reward:", np.mean(reward_log))

        return np.concatenate(tuple(obs), axis=0),\
            np.concatenate(tuple(action), axis=0), \
            np.concatenate(tuple(rew), axis=0), \
            np.concatenate(tuple(obs_next), axis=0), \
            np.concatenate(tuple(done), axis=0)

    def update_params(self, _obs, _action, _rew, _obs_next, _done):
        logger.debug("update parms")
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
        # policy_loss 表示 value 对action的评分值（Q值），评分越大越好
        loss_a = torch.mul(-1, torch.mean(self.value.forward(obs, action_new)))
        loss_reg = torch.mean(torch.pow(model_out, 2)) * 1e-2
        policy_loss = loss_reg + loss_a

        self.policy_opt.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(),
                                 self.args.max_grad_norm)
        self.policy_opt.step()

        soft_update(self.target_policy, self.policy, self.args.tau)
        soft_update(self.target_value, self.value, self.args.tau)
        return value_loss, policy_loss, loss_reg

    def learn(self):
        logger.info("learning")
        logger.info("warmming up: %d" % self.args.warm_up)
        self.explore(self.args.warm_up, self.args.sample_size)
        logger.info("warm up: %d" % self.args.warm_up)
        for i_iter in range(self.args.max_iter_num):
            obs, action, rew, obs_next, done = self.explore(
                explore_size=self.args.explore_size,
                sample_size=self.args.sample_size)
            value_loss, policy_loss, loss_reg = self.update_params(
                obs, action, rew, obs_next, done)
            print(value_loss, policy_loss, loss_reg)

            if (i_iter + 1) % self.args.log_interval == 0:
                # TODO: avg_reward
                avg_reward = 0.0
                logger.info("iter=%d, avg_reward=%.3f" % (i_iter, avg_reward))

            if (i_iter + 1) % self.args.save_model_interval == 0:
                self.save(self.model_dir)

            """clean up gpu memory"""
            clear_memory()


def main():
    # import logging
    # logger.setLevel(logging.DEBUG)
    args = common_arg_parser()
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
                  env.n, opt_fn, args)
    agent.learn()


if __name__ == '__main__':
    main()
