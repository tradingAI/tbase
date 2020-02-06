# -*- coding:utf-8 -*-

import os
import random

import numpy as np

import tgym


def set_global_seeds(seed):
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass
    np.random.seed(seed)
    random.seed(seed)


def make_env(env_id, seed, args):
    """
    Create a wrapped, monitored gym.Env for Tgym.
    """
    set_global_seeds(seed)
    ts_token = os.getenv("TUSHARE_TOKEN")
    codes = args.codes.split(",")
    indexs = args.indexs.split(",")

    m = tgym.market.Market(
            ts_token=ts_token,
            start=args.start,
            end=args.end,
            codes=codes,
            indexs=indexs,
            data_dir=args.data_dir)
    env = tgym.scenario.make_env(scenario=args.scenario,
                                 market=m,
                                 investment=args.investment,
                                 look_back_days=args.look_back_days,
                                 )
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
    parser.add_argument("--indexs", type=str, default="000001.SH,399001.SZ",
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
