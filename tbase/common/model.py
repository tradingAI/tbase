# -*- coding:utf-8 -*-
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class AbstractAgent(nn.Module):
    def __init__(self):
        super(AbstractAgent, self).__init__()

    def act(self, input):
        # flow the input through the nn
        policy, value = self.forward(input)
        return policy, value
