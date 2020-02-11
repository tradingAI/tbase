# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable

from tbase.common.torch_utils import fc, lstm
from tbase.network.base import BaseNet


class LSTM_Merge_MLP(BaseNet):
    def __init__(self, seq_len=11, obs_input_size=10, rnn_hidden_size=300,
                 num_layers=1, dropout=0.0, learning_rate=0.001,
                 act_input_size=4, act_fc1_size=200,
                 act_fc2_size=100, output_size=1,  activation=None):
        super(LSTM_Merge_MLP, self).__init__()
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

        if activation is None:
            self.activation = nn.Tanh()
        else:
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
        # actor full connect
        act_output = self.fc1(act_n)
        act_output = self.fc2(act_output)
        fc_input = torch.cat((output, act_output), dim=1)
        v = self.fc3(fc_input)
        return v


def get_value_net(env, args):
    if args.value_net == "LSTM_Merge_MLP":
        seq_len = args.look_back_days
        input_size = env.input_size
        act_size = env.action_space
        # TODO:
        return LSTM_Merge_MLP(
            seq_len=seq_len, obs_input_size=input_size, rnn_hidden_size=300,
            num_layers=1, dropout=0.0, learning_rate=0.001,
            act_input_size=act_size,
            act_fc1_size=200, act_fc2_size=100, output_size=1, activation=None)
    else:
        raise ValueError("Not implement value_net: %s" % args.value_net)
