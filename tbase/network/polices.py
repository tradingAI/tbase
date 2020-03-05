# -*- coding:utf-8 -*-

import numpy as np
import torch
from torch.autograd import Variable
from torch.distributions import Normal

from tbase.common.random_process import OrnsteinUhlenbeckProcess
from tbase.common.torch_utils import default_device, fc, get_activation, lstm
from tbase.network.base import BaseNet, BasePolicy


class RandomPolicy(BasePolicy):
    def __init__(self, action_size):
        super(RandomPolicy, self).__init__()
        self.action_size = action_size
        self.random_process = OrnsteinUhlenbeckProcess(0.1, size=action_size)
        mean = torch.Tensor([0] * action_size)
        self.normal = Normal(mean, 1)

    def action(self, obs):
        return self.normal.sample()

    def select_action(self, obs):
        action = self.action(obs).numpy() + self.random_process.sample()
        action = np.clip(action, self.action_low, self.action_high)
        return action


class LSTM_MLP(BasePolicy):
    def __init__(self, device=None, seq_len=11, input_size=10, hidden_size=300,
                 output_size=4, num_layers=1, dropout=0.0, learning_rate=0.001,
                 fc_size=200, activation=None, ou_theta=0.15,
                 ou_sigma=0.2, ou_mu=0):
        super(LSTM_MLP, self).__init__(device)
        self.seq_len = seq_len
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        # learning rate
        self.learning_rate = learning_rate
        # 定义和初始化网络
        self.rnn = lstm(input_size, hidden_size, num_layers, dropout)
        self.fc1 = fc(hidden_size, fc_size)
        self.fc2 = fc(fc_size, output_size)
        self.activation = activation
        self.random_process = OrnsteinUhlenbeckProcess(
            size=output_size, theta=ou_theta, mu=ou_mu, sigma=ou_sigma)

    def init_hidden(self, batch_size):
        h_0 = Variable(torch.randn(self.num_layers, batch_size,
                       self.hidden_size)).to(self.device, torch.float)
        c_0 = Variable(torch.randn(self.num_layers, batch_size,
                       self.hidden_size)).to(self.device, torch.float)
        return h_0, c_0

    def action(self, obs, with_reg=False):
        # obs: seq_len, batch_size, input_size
        h_0, c_0 = self.init_hidden(obs.shape[1])
        output, _ = self.rnn(obs.to(self.device), (h_0, c_0))
        output = self.activation(output)
        encoded = self.activation(self.fc1(output[-1, :, :]))
        action = torch.tanh(self.fc2(encoded))
        if with_reg:
            return action, encoded
        return action

    def select_action(self, obs):
        # obs: seq_len, batch_size, input_size
        action = self.action(obs, with_reg=False)
        action = action.detach().cpu()[0].numpy()
        action += self.random_process.sample()
        action = np.clip(action, self.action_low, self.action_high)
        return action


# TODO: refactor
class LSTM_MLP_A2C(BaseNet):
    def __init__(self, device=None, seq_len=11, input_size=10, hidden_size=300,
                 output_size=4, num_layers=1, dropout=0.0, learning_rate=0.001,
                 fc_size=200, activation=None):
        super(LSTM_MLP_A2C, self).__init__(device)
        self.seq_len = seq_len
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        # learning rate
        self.learning_rate = learning_rate
        # 定义和初始化网络
        self.rnn = lstm(input_size, hidden_size, num_layers, dropout)
        self.fc1 = fc(hidden_size, fc_size)
        self.fc2 = fc(fc_size, output_size)
        self.fc3 = fc(fc_size, output_size)
        self.activation = activation
        self.dist = Normal
        self.count = 1000

    def init_hidden(self, batch_size):
        h_0 = Variable(torch.randn(self.num_layers, batch_size,
                       self.hidden_size)).to(self.device, torch.float)
        c_0 = Variable(torch.randn(self.num_layers, batch_size,
                       self.hidden_size)).to(self.device, torch.float)
        return h_0, c_0

    def forward(self, obs, explore=False, act=None):
        # obs: seq_len, batch_size, input_size
        h_0, c_0 = self.init_hidden(obs.shape[1])
        output, _ = self.rnn(obs.to(self.device), (h_0, c_0))
        output = self.activation(output)
        encoded = self.activation(self.fc1(output[-1, :, :]))
        mu = torch.tanh(self.fc2(encoded))
        # Note: sigma = sigma + max(1000. / self.count, 0.1) is better
        sigma = torch.nn.functional.softplus(self.fc3(encoded))

        self.count += 1
        dist = self.dist(mu, sigma)
        action = dist.sample()
        action = torch.clamp(action, self.action_low, self.action_high)
        if explore:
            return action

        log_prob = dist.log_prob(act)
        entropy = dist.entropy()
        return action, entropy, log_prob, sigma.mean()

    def action(self, obs):
        # obs: seq_len, batch_size, input_size
        h_0, c_0 = self.init_hidden(obs.shape[1])
        output, _ = self.rnn(obs.to(self.device), (h_0, c_0))
        output = self.activation(output)
        encoded = self.activation(self.fc1(output[-1, :, :]))
        mu = torch.tanh(self.fc2(encoded))
        return mu


def get_policy_net(env, args):
    seq_len = args.look_back_days
    input_size = env.input_size
    act_size = env.action_space
    activation = get_activation(args.activation)
    device = default_device
    if not(args.device is None):
        device = args.device
    if args.policy_net == "LSTM_MLP":
        return LSTM_MLP(
            device=device, seq_len=seq_len,
            input_size=input_size, hidden_size=300,
            output_size=act_size, num_layers=1, dropout=0.0,
            learning_rate=args.lr, fc_size=200, activation=activation,
            ou_theta=0.15, ou_sigma=0.2, ou_mu=0).to(device)
    elif args.policy_net == "Random":
        return RandomPolicy(act_size)
    elif args.policy_net == "LSTM_MLP_A2C":
        return LSTM_MLP_A2C(
            device=device, seq_len=seq_len,
            input_size=input_size, hidden_size=300,
            output_size=act_size, num_layers=1, dropout=0.0,
            learning_rate=args.lr, fc_size=200,
            activation=activation).to(device)
    else:
        raise ValueError("Not implement policy_net: %s" % args.value_net)
