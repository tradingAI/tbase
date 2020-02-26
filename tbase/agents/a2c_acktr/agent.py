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
        pass
