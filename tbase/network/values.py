# -*- coding:utf-8 -*-

import torch
from torch.autograd import Variable

from tbase.common.torch_utils import default_device, fc, get_activation, lstm
from tbase.network.base import BaseNet


class DoubleValue(BaseNet):
    def __init__(self, net1, net2):
        super(DoubleValue, self).__init__()
        self.net1 = net1
        self.net2 = net2

    def forward(self, state, action):
        q1 = self.net1.forward(state, action)
        q2 = self.net2.forward(state, action)
        return q1, q2

    def Q1(self, state, action):
        return self.net1.forward(state, action)


class LSTM_Merge_MLP(BaseNet):
    def __init__(self, device=None, seq_len=11, obs_input_size=10,
                 rnn_hidden_size=300, num_layers=1, dropout=0.0,
                 learning_rate=0.001,
                 act_input_size=4, act_fc1_size=200,
                 act_fc2_size=100, output_size=1,  activation=None):
        super(LSTM_Merge_MLP, self).__init__(device)
        self.seq_len = seq_len
        self.obs_input_size = obs_input_size
        self.rnn_hidden_size = rnn_hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size
        # learning rate
        self.learning_rate = learning_rate
        # 定义和初始化网络
        self.rnn = lstm(obs_input_size, rnn_hidden_size, num_layers, dropout)
        self.fc1 = fc(act_input_size, act_fc1_size)
        self.fc2 = fc(act_fc1_size, act_fc2_size)
        self.fc3 = fc(rnn_hidden_size + act_fc2_size, output_size)
        self.activation = activation

    def init_hidden(self, batch_size):
        h_0 = Variable(torch.randn(self.num_layers, batch_size,
                       self.rnn_hidden_size)).to(self.device, torch.float)
        c_0 = Variable(torch.randn(self.num_layers, batch_size,
                       self.rnn_hidden_size)).to(self.device, torch.float)
        return h_0, c_0

    def forward(self, obs, act_n):
        # obs(seq_len, batch_size, obs_input_size)
        h_0, c_0 = self.init_hidden(obs.shape[1])
        # output (seq_len, batch_size, hidden_size * num_directions)
        output, _ = self.rnn(obs, (h_0, c_0))
        # 取最后的一个输出
        output = output[-1, :, :].view(obs.shape[1], self.rnn_hidden_size)
        output = self.activation(output)
        # actor full connect
        act_output = self.activation(self.fc1(act_n))
        act_output = self.activation(self.fc2(act_output))
        fc_input = torch.cat((output, act_output), dim=1)
        v = self.fc3(fc_input)
        return v


# TODO: refactor
class LSTM_MLP_A2C(BaseNet):
    def __init__(self, device=None, seq_len=11, input_size=10, hidden_size=300,
                 num_layers=1, dropout=0.0, learning_rate=0.001,
                 fc_size=200, output_size=1, activation=None):
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
        self.activation = activation

    def init_hidden(self, batch_size):
        h_0 = Variable(torch.randn(self.num_layers, batch_size,
                       self.hidden_size)).to(self.device, torch.float)
        c_0 = Variable(torch.randn(self.num_layers, batch_size,
                       self.hidden_size)).to(self.device, torch.float)
        return h_0, c_0

    def forward(self, obs):
        # obs: seq_len, batch_size, input_size
        h_0, c_0 = self.init_hidden(obs.shape[1])
        output, _ = self.rnn(obs.to(self.device), (h_0, c_0))
        output = self.activation(output)
        output = self.activation(self.fc1(output[-1, :, :]))
        v = self.fc2(output)
        return v


def get_single_value_net(env, args):
    seq_len = args.look_back_days
    input_size = env.input_size
    act_size = env.action_space
    activation = get_activation(args.activation)
    device = default_device
    if not(args.device is None):
        device = args.device
    if args.value_net == "LSTM_Merge_MLP":
        return LSTM_Merge_MLP(
            device=device,
            seq_len=seq_len, obs_input_size=input_size, rnn_hidden_size=300,
            num_layers=1, dropout=0.0, learning_rate=args.lr,
            act_input_size=act_size,
            act_fc1_size=200, act_fc2_size=100, output_size=1,
            activation=activation).to(device)
    if args.value_net == "LSTM_MLP_A2C":
        return LSTM_MLP_A2C(
            device=device,
            seq_len=seq_len, input_size=input_size, hidden_size=300,
            num_layers=1, dropout=0.0, learning_rate=args.lr,
            fc_size=200, output_size=1,
            activation=activation).to(device)
    else:
        raise ValueError("Not implement value_net: %s" % args.value_net)


def get_double_value_net(env, args):
    net1 = get_single_value_net(env, args)
    net2 = get_single_value_net(env, args)
    net = DoubleValue(net1, net2)
    return net


def get_value_net(env, args):
    if args.alg == 'td3':
        return get_double_value_net(env, args)
    return get_single_value_net(env, args)
