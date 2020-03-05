# -*- coding:utf-8 -*-

import random
import time

import numpy as np
import torch
from torch import nn

from tbase.agents.base.ac_agent import ACAgent
from tbase.common.logger import logger
from tbase.common.torch_utils import clear_memory


class Agent(ACAgent):
    def __init__(self, env=None, args=None):
        super(Agent, self).__init__(env, args)

    # TODO: add multi-process explore method
    def explore(self, env, state, size, print_actions):
        t_start = time.time()
        num_steps = 0
        portfolios = []
        states, actions, rewards, next_states, dones = [], [], [], [], []
        while num_steps < size:
            state_var = torch.tensor(state).unsqueeze(0).permute(1, 0, 2).to(
                torch.float)
            with torch.no_grad():
                action = self.policy.forward(state_var, True)
                action = action.detach().cpu()[0].numpy().astype(np.float)
            if print_actions:
                if random.random() < 0.01:
                    print("tbase.agents.ddpg.agent action:" + str(action))
            next_state, reward, done, info, _ = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            num_steps += 1

            if done:
                state = env.reset()
                portfolios.append(info["portfolio_value"])
                break
            state = next_state
        e_t = time.time() - t_start
        return np.array(states), np.array(actions), np.array(rewards), \
            np.array(next_states), dones, portfolios, e_t

    def update_params(self, _obs, _action, _rew, _obs_next, _done, iter):
        t_start = time.time()
        # value
        rewards = torch.tensor(_rew, device=self.policy.device,
                               dtype=torch.float)
        rewards = rewards.reshape(-1, 1)
        states = torch.from_numpy(_obs).permute(1, 0, 2).to(
            self.policy.device, torch.float)
        values = self.value.forward(states)
        # R_t: Return from time step t with discount factor gamma
        R = torch.zeros(len(rewards) + 1, 1)
        if not _done[-1]:
            R[-1] = values[-1]
        value_loss = 0
        for i in reversed(range(len(rewards))):
            R[i] = self.args.gamma * R[i + 1] + rewards[i]
        advantages = R[:-1] - values
        value_loss = advantages.pow(2).mean()
        self.value_opt.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value.parameters(),
                                 self.args.max_grad_norm)
        self.value_opt.step()
        # policy
        actions = torch.from_numpy(_action).to(self.policy.device, torch.float)
        actions = actions.reshape(actions.shape[0], -1)
        n_action, action_log_probs, dist_entropy, sigma = self.policy.forward(
            states, False, actions)
        self.writer.add_scalar('action/sigma', sigma, iter)

        log_prob = (advantages.detach() * action_log_probs).mean()
        abs_log_prob = torch.abs(advantages.detach() * action_log_probs).mean()
        self.writer.add_scalar('action/abs_log_prob', abs_log_prob, iter)
        action_reg = torch.mean(torch.pow(n_action, 2))
        self.writer.add_scalar('action/reg', action_reg, iter)
        dist_entropy = dist_entropy.mean() * self.args.entropy_coef

        action_loss = - log_prob - dist_entropy

        # if self.acktr: TODO
        self.policy_opt.zero_grad()
        action_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(),
                                 self.args.max_grad_norm)
        self.policy_opt.step()

        ut = time.time() - t_start
        self.writer.add_scalar('time/update', ut, iter)
        self.writer.add_scalar('loss/value', value_loss, iter)
        self.writer.add_scalar('loss/policy', action_loss, iter)
        self.writer.add_scalar('action/dist_entropy',
                               dist_entropy, iter)

        return value_loss, action_loss, dist_entropy, ut

    def learn(self):
        logger.info("learning started")
        i = 0
        current_portfolio = 1.0
        t_start = time.time()
        state = self.envs[0].reset()
        for i_iter in range(self.args.max_iter_num):
            obs, act, rew, obs_t, done, ports, e_t = \
                self.explore(
                    self.envs[0],
                    state,
                    self.args.t_max,
                    self.args.print_action)
            state = obs[-1]
            for p in ports:
                i += 1
                self.writer.add_scalar('reward/portfolio', p, i)
                current_portfolio = p
                if current_portfolio > self.best_portfolio:
                    self.best_portfolio = current_portfolio
                    logger.info("iter: %d, new best portfolio: %.3f" % (
                        i_iter + 1, self.best_portfolio))
                    self.save(self.model_dir)
            self.writer.add_scalar('time/explore', e_t, i_iter)
            self.writer.add_scalar('reward/policy', np.mean(rew), i_iter)

            self.update_params(obs, act, rew, obs_t, done, i_iter)

            if (i_iter + 1) % self.args.log_interval == 0:
                msg = "total update time: %.1f secs" % (time.time() - t_start)
                msg += ", current_portfolio: %.3f" % current_portfolio
                logger.info(msg)
            clear_memory()

        logger.info("Final best portfolio: %.3f" % self.best_portfolio)
        self.save_best_portofolio(self.model_dir)
