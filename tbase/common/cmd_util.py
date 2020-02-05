# -*- coding:utf-8 -*-

import random

import numpy as np

import tgym


def set_global_seeds(i):
    try:
        import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0

    myseed = i + 1000 * rank if i is not None else None
    try:
        import torch
        torch.manual_seed(myseed)
        torch.cuda.manual_seed(myseed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass
    np.random.seed(myseed)
    random.seed(myseed)


def make_trade_env(env_id,
                   seed,
                   rank=0,
                   ts_token=None,
                   start="20190101",
                   end="20200101",
                   codes=["000001.SZ", "000002.SZ"],
                   indexs=["000001.SH", "399001.SZ"],
                   data_dir="/tmp/tgym",
                   scenario="multi_vol"):
    """
    Create a wrapped, monitored gym.Env for Tgym.
    """
    set_global_seeds(seed)
    m = tgym.market.Market(
            ts_token=ts_token,
            start=start,
            end=end,
            codes=codes,
            indexs=indexs,
            data_dir=data_dir)
    env = tgym.scenario.make_env(scenario=scenario,
                                 market=m,
                                 investment=100000.0,
                                 look_back_days=10)
    return env


def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    # 环境
    parser.add_argument('--env', help='environment scenario', type=str,
                        default='multi_vol')
    parser.add_argument("--codes", type=str, default="000001.SZ,000002.SZ",
                        help="tushare code of the experiment stocks")
    parser.add_argument("--indexs", type=str, default="",
                        help="tushare code of the indexs")
    parser.add_argument("--start", type=str, default='20190101',
                        help="when start the game")
    parser.add_argument("--end", type=str, default='20191231',
                        help="when end the game")
    parser.add_argument("--invesment", type=float, default=100000,
                        help="the invesment for each stock")
    parser.add_argument("--look_back_days", type=int, default=10,
                        help="how many days shoud look back")
    parser.add_argument('--num_env', default=None, type=int,
                        help='Number of environment copies run in parallel.')
    # 训练参数
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--alg', help='Algorithm', type=str, default='ppo2')
    parser.add_argument('--max_episode', type=float, default=1000)
    # 模型参数
    parser.add_argument('--policy-net', default=None,
                        help='network type (mlp, lstm, cnn_lstm)')
    parser.add_argument('--value-net', default=None,
                        help='network type (mlp, lstm_mpl)')
    parser.add_argument('--save_path', help='Path to save trained model to',
                        default=None, type=str)
    parser.add_argument('--log_path', default=None, type=str,
                        help='Directory to save learning curve data.')
    # 运行参数
    parser.add_argument('--play', default=False, action='store_true')
    return parser
