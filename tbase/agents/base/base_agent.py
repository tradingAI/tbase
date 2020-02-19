import torch.nn as nn


class BaseAgent(nn.Module):
    def __init__(self, env, args, *other_args):
        super(BaseAgent, self).__init__()
        self.args = args

    def learn(self, *args):
        raise NotImplementedError
