# -*- coding:utf-8 -*-
import torch.nn as nn

from tbase.common.torch_utils import device


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.device = device
        self.action_low = -1
        self.action_high = 1

    def forward(*args):
        raise NotImplementedError


class BasePolicy(BaseNet):
    def __init__(self):
        super(BasePolicy, self).__init__()

    def action(self, *args):
        raise NotImplementedError

    def select_action(self, *args):
        raise NotImplementedError
