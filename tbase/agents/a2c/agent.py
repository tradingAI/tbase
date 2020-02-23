# -*- coding:utf-8 -*-

import time

import torch

from tbase.agents.base.ac_agent import ACAgent
from tbase.common.logger import logger
from tbase.common.torch_utils import clear_memory


class Agent(ACAgent):
    def __init__(self, env=None, args=None):
        super(Agent, self).__init__(env, args)

    def update_params(self, _obs, _action, _rew, _obs_next, _done):
        # TODO
        pass

    def learn(self):
        if self.args.num_env > 1:
            self.policy.share_memory()
        self.warm_up()
        logger.info("learning started")
        i = 0
        current_portfolio = 1.0
        t_start = time.time()
        for i_iter in range(self.args.max_iter_num):
            obs, act, rew, obs_t, done, avg_reward, e_t, ports = [None] * 8
            if self.args.num_env == 1:
                obs, act, rew, obs_t, done, avg_reward, e_t, ports = \
                    self.simple_explore()
            else:
                obs, act, rew, obs_t, done, avg_reward, e_t, ports = \
                    self.explore()
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
                v_loss, p_loss, p_reg, act_reg, u_t = self.update_params(
                    obs, act, rew, obs_t, done)
            except Exception as error:
                print(error)
            self.writer.add_scalar('time/update', u_t, i_iter)
            self.writer.add_scalar('loss/value', v_loss, i_iter)
            self.writer.add_scalar('loss/policy', p_loss, i_iter)
            self.writer.add_scalar('reg/action', act_reg, i_iter)
            self.writer.add_scalar('reg/policy', p_reg, i_iter)
            self.writer.add_scalar('reward/policy',
                                   torch.tensor(avg_reward), i_iter)

            if (i_iter + 1) % self.args.log_interval == 0:
                msg = "total update time: %.1f secs" % (time.time() - t_start)
                msg += ", iter=%d, avg_reward=%.3f" % (i_iter + 1, avg_reward)
                msg += ", current_portfolio: %.3f" % current_portfolio
                logger.info(msg)
            clear_memory()

        self.writer.close()
        logger.info("Final best portfolio: %.3f" % self.best_portfolio)
        self.save_best_portofolio(self.model_dir)
