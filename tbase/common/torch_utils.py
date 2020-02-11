import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def opt_rmsprop(params, lr):
    return torch.optim.RMSprop(params, lr)


def opt_fn(name="RMSprop"):
    if name == "RMSprop":
        return opt_rmsprop
    else:
        raise Exception(
            "tbase.common.torch_utils opt_fn: NotImplementedError %s " % name)


def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - tau) * target_param.data + tau * source_param.data)


def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
