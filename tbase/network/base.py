# -*- coding:utf-8 -*-
import torch.nn as nn


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

    def forward(*args):
        raise NotImplementedError


# Actor Critic Agent
class ACAgent(nn.Module):
    def __init__(self):
        super(ACAgent, self).__init__()

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def action(state):
        raise NotImplementedError

    def select_action(state):
        raise NotImplementedError

    def learn(*args):
        raise NotImplementedError
