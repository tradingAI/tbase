# -*- coding:utf-8 -*-

import random
import time

import numpy as np
import torch

from tbase.agents.base.ac_agent import ACAgent
from tbase.common.logger import logger
from tbase.common.torch_utils import clear_memory


class Agent(ACAgent):
    def __init__(self, env=None, args=None):
        super(Agent, self).__init__(env, args)

    def explore(self, env, state, size, print_actions):
        t_start = time.time()
        num_steps = 0
        portfolios = []
        states, actions, rewards, next_states, dones = [], [], [], [], []
        while num_steps < size:
            state_var = torch.tensor(state).unsqueeze(0).permute(1, 0, 2).to(
                torch.float)
            with torch.no_grad():
                action = self.policy.action(state_var, True)
                action = action.astype(np.float)
            if print_actions:
                if random.random() < 0.001:
                    print("tbase.agents.ddpg.agent action:" + str(action))
            next_state, reward, done, info, _ = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            rewards.append(reward)
            num_steps += 1

            if done:
                state = env.reset()
                portfolios.append(info["portfolio_value"])
                break
            state = next_state
        e_t = time.time() - t_start
        return states, actions, rewards, next_states, dones, portfolios, e_t

    def update_params(self, states, actions, rewards, next_states, dones):
        t_start = time.time()

        action, action_log_probs, dist_entropy = self.policy.forward(states)
        values = self.value.forward(states, action)
        R = torch.zeros(1, 1)
        if dones[-1] != 1:
            R = values[-1]

        value_loss = 0
        advantages = []
        for i in reversed(range(len(rewards))):
            R = self.args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + advantage.pow(2)
            advantages.append(advantage)

        value_loss = advantages.pow(2).mean()
        action_loss = -(advantages.detach() * action_log_probs).mean()

        if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
            # Compute fisher, see Martens 2014
            self.actor_critic.zero_grad()
            pg_fisher_loss = - action_log_probs.mean()

            value_noise = torch.randn(values.size())
            if values.is_cuda:
                value_noise = value_noise.cuda()

            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

            fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False

        self.optimizer.zero_grad()
        (value_loss * self.value_loss_coef + action_loss -
         dist_entropy * self.entropy_coef).backward()
        self.optimizer.step()

        ut = time.time() - t_start
        return value_loss.item(), action_loss.item(), dist_entropy.item(), ut

    def learn(self):
        if self.args.num_env > 1:
            self.policy.share_memory()
        self.warm_up()
        logger.info("learning started")
        i = 0
        current_portfolio = 1.0
        t_start = time.time()
        for i_iter in range(self.args.max_iter_num):
            obs, act, rew, obs_t, done, ports, e_t = \
                self.explore(self.args.t_max)

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
            try:
                v_loss, p_loss, dist_entropy, u_t = self.update_params(
                    obs, act, rew, obs_t, done)
            except Exception as error:
                print(error)
            self.writer.add_scalar('time/update', u_t, i_iter)
            self.writer.add_scalar('loss/value', v_loss, i_iter)
            self.writer.add_scalar('loss/policy', p_loss, i_iter)
            self.writer.add_scalar('dist_entropy/action', dist_entropy, i_iter)

            if (i_iter + 1) % self.args.log_interval == 0:
                msg = "total update time: %.1f secs" % (time.time() - t_start)
                msg += ", current_portfolio: %.3f" % current_portfolio
                logger.info(msg)
            clear_memory()

        logger.info("Final best portfolio: %.3f" % self.best_portfolio)
        self.save_best_portofolio(self.model_dir)
