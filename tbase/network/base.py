# -*- coding:utf-8 -*-
import torch.nn as nn

from tbase.common.torch_utils import device


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.device = device

    def forward(*args):
        raise NotImplementedError


class BasePolicy(BaseNet):
    def __init__(self):
        super(BasePolicy, self).__init__()

    def action(self):
        raise NotImplementedError

    def select_action(self):
        raise NotImplementedError
