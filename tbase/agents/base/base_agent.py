import os
from datetime import datetime

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class BaseAgent(nn.Module):
    def __init__(self, env, args, *other_args):
        super(BaseAgent, self).__init__()
        self.args = args
        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        module_name = self.get_module_name()
        log_dir = os.path.join(args.tensorboard_dir, module_name, TIMESTAMP)
        self.writer = SummaryWriter(log_dir)
        self.best_portfolio = -1.0
        self.run_id = args.run_id
        self.name = self.get_agent_name()
        self.model_dir = self.get_model_dir()

    def learn(self, *args):
        raise NotImplementedError

    def save_best_portofolio(self, dir):
        best_portfolio_path = os.path.join(dir, "best_portfolios.txt")
        f = open(best_portfolio_path, "a")
        msg = "=" * 80 + "\n"
        msg += "best_portfolio: " + str(self.best_portfolio) + "\n"
        msg += str(self.args) + "\n"
        msg += datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n"

        f.write(msg)
        f.close()

    def print_net(self, net):
        for param_tensor in net.state_dict():
            print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    def get_module_name(self):
        module_splits = self.__class__.__module__.split(".")
        return module_splits[-2]

    def get_agent_name(self):
        code_str = self.args.codes.replace(",", "_")
        name = self.get_module_name() + "_" + code_str
        return name

    def get_model_dir(self):
        dir = os.path.join(self.args.model_dir, self.name)
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir

    def eval(self, *args):
        raise NotImplementedError
