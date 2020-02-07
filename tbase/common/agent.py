# -*- coding:utf-8 -*-
import os

import torch
import torch.nn as nn


# Actor Critic Agent
class ACAgent(nn.Module):
    def __init__(self, policy_net, value_net, *args):
        super(ACAgent, self).__init__()
        # policy net
        self.policy = policy_net
        # value net
        self.value = value_net

    def save(self, dir):
        torch.save(
            self.policy.state_dict(),
            '{}/{}.policy.pkl'.format(dir, self.name)
        )
        torch.save(
            self.value.state_dict(),
            '{}/{}.critic.pkl'.format(dir, self.name)
        )

    def load(self, dir):
        if dir is None or not os.path.exist(dir):
            raise "dir is invalid"
        self.policy.load_state_dict(
            torch.load('{}/{}.policy.pkl'.format(dir, self.name))
        )
        self.value.load_state_dict(
            torch.load('{}/{}.value.pkl'.format(dir, self.name))
        )

    # 探索与搜集samples
    def explore(self, explore_size, sample_size):
        raise NotImplementedError

    def learn(*args):
        raise NotImplementedError
