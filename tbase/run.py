# -*- coding:utf-8 -*-
"""
1. 提供多进程并环境
2. 根据输入选择：算法，Policy-Net, Value-Net, 优化算法. 然后生成Agent
3. 训练保存模型performance, 参数
"""
import torch

from tbase.agents.ddpg.agent import Agent as ddpg
from tbase.common.cmd_util import common_arg_parser, make_env
from tbase.common.logger import logger
from tbase.network.polices import get_policy_net
from tbase.network.values import get_value_net


def opt_fn(params, lr):
    return torch.optim.RMSprop(params, lr)


def get_agent(env, args):

    policy_net = get_policy_net(env, args)
    target_policy_net = get_policy_net(env, args)
    value_net = get_value_net(env, args)
    target_value_net = get_value_net(env, args)
    if args.alg == "ddpg":
        return ddpg(
            policy_net, value_net,
            target_policy_net, target_value_net,
            env.n, opt_fn, args.tensorboard_dir, args)


def main():
    args = common_arg_parser()
    if args.debug:
        import logging
        logger.setLevel(logging.DEBUG)
    env = make_env(args=args)
    print("\n" + "*" * 80) 
    logger.info("Initializing agent by parameters:")
    logger.info(str(args))
    agent = get_agent(env, args)
    logger.info("Training agent")
    agent.learn()
    logger.info("Train finished, see details by run tensorboard --logdir=%s" %
                args.tensorboard_dir)


if __name__ == '__main__':
    main()
