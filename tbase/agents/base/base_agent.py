import os
from datetime import datetime

import torch.nn as nn


class BaseAgent(nn.Module):
    def __init__(self, env, args, *other_args):
        super(BaseAgent, self).__init__()
        self.args = args

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
