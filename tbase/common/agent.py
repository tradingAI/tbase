# -*- coding:utf-8 -*-
import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from tbase.common.torch_utils import opt_fn
from tbase.network.polices import get_policy_net
from tbase.network.values import get_value_net


# Actor Critic Agent
class ACAgent(nn.Module):
    def __init__(self, env, args, *other_args):
        super(ACAgent, self).__init__()
        policy_net = get_policy_net(env, args)
        target_policy_net = get_policy_net(env, args)
        value_net = get_value_net(env, args)
        target_value_net = get_value_net(env, args)
        optimizer_fn = opt_fn(args.opt_fn)
        # policy net
        self.policy = policy_net
        self.target_policy = target_policy_net
        self.policy_opt = optimizer_fn(
            params=filter(lambda p: p.requires_grad, self.policy.parameters()),
            lr=self.policy.learning_rate)
        # value net
        self.value = value_net
        self.target_value = target_value_net
        self.value_opt = optimizer_fn(
            params=filter(lambda p: p.requires_grad, self.value.parameters()),
            lr=self.value.learning_rate)
        self.writer = SummaryWriter(log_dir=args.tensorboard_dir)

    def save(self, dir):
        torch.save(
            self.policy.state_dict(),
            '{}/{}.policy.pkl'.format(dir, self.name)
        )
        torch.save(
            self.value.state_dict(),
            '{}/{}.value.pkl'.format(dir, self.name)
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
