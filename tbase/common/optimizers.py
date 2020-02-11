import torch

mapping = {}


def register(name):
    def _thunk(func):
        mapping[name] = func
        return func
    return _thunk


@register("rmsprop")
def rmsprop():
    return torch.optim.RMSprop


@register("adam")
def adam():
    return torch.optim.Adam


def get_optimizer_func(name):
    """
    If you want to register your own optimizer function, you just need:
    Usage Example:
    -------------
    from tbase.common.optimizers import register
    @register("your_reward_function_name")
    def your_optimizer_func(**kwargs):
        ...
        return optimizer_func
    """
    if callable(name):
        return name
    elif name in mapping:
        return mapping[name]
    else:
        raise ValueError('Unknown optimizer_func: {}'.format(name))
