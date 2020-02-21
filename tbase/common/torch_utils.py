import gc

import torch
import torch.nn as nn

default_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# full connect layer
def fc(in_features, out_features):
    layer = nn.Linear(in_features, out_features)
    nn.init.xavier_normal_(layer.weight)
    nn.init.constant_(layer.bias, 0)
    return layer


def lstm(input_size, hidden_size, num_layers, dropout):
    rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                  num_layers=num_layers, dropout=dropout)
    return rnn


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def to_device(device, *args):
    return [x.to(device) for x in args]


def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - tau) * target_param.data + tau * source_param.data)


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_activation(activation):
    if activation == "tanh":
        return nn.Tanh()
    elif activation == "relu":
        return nn.ReLU()
    else:
        raise NotImplementedError
