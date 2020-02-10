# -*- coding:utf-8 -*-

import os
import random

import numpy as np

from tgym.market import Market
from tgym.scenario import make_env as _make_env


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


def make_env(args):
    """
    Create a wrapped, monitored gym.Env for Tgym.
    """
    ts_token = os.getenv("TUSHARE_TOKEN")
    codes = args.codes.split(",")
    indexs = args.indexs.split(",")

    m = Market(
            ts_token=ts_token,
            start=args.start,
            end=args.end,
            codes=codes,
            indexs=indexs,
            data_dir=args.data_dir)
    used_infos = ["equities_hfq_info", "indexs_info"]
    env = _make_env(
        scenario=args.scenario,
        market=m,
        investment=args.investment,
        look_back_days=args.look_back_days,
        used_infos=used_infos,
        reward_fn=args.reward_fn)
    return env


def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    import argparse
    parser = argparse.ArgumentParser("reinforcement learning trade agents")
    # 环境
    parser.add_argument('--scenario', help='environment scenario', type=str,
                        default='average')
    parser.add_argument("--codes", type=str, default="000001.SZ",
                        help="tushare code of the experiment stocks")
    parser.add_argument(
        "--indexs", type=str, default="000001.SH,399001.SZ",
        help="tushare code of the indexs, 000001.SH:沪指, \
        399001.SZ: 深指, detal:https://tushare.pro/document/2?doc_id=94")
    parser.add_argument("--start", type=str, default='20190101',
                        help="when start the game")
    parser.add_argument("--end", type=str, default='20191231',
                        help="when end the game")
    parser.add_argument("--investment", type=float, default=100000,
                        help="the investment for each stock")
    parser.add_argument("--look_back_days", type=int, default=10,
                        help="how many days shoud look back")
    parser.add_argument("--data_dir", type=str, default='/tmp/tgym',
                        help="directory for tgym store trade data")
    parser.add_argument('--num_env', default=4, type=int,
                        help='Number of environment copies run in parallel.')
    # 模型参数
    parser.add_argument('--policy_net', default='LSTM_MLP',
                        help='network type (LSTM_MLP)')
    parser.add_argument('--value_net', default='LSTM_Merge_MLP',
                        help='network type (LSTM_Merge_MLP)')
    parser.add_argument('--reward_fn', default="daily_return_add_price_bound",
                        help='reward function')
    # 训练参数
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--alg', help='Algorithm', type=str, default='ddpg')
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="discount factor")
    parser.add_argument("--max_grad_norm", type=float, default=5,
                        help="max gradient norm for clip")
    parser.add_argument("--tau", type=int, default=0.95,
                        help="how depth we exchange the parameters of the nn")
    parser.add_argument('--explore_size', type=int, default=400)
    parser.add_argument('--sample_size', type=int, default=200)
    parser.add_argument('--warm_up', type=int, default=10000)

    parser.add_argument('--max-iter-num', type=int, default=10000, metavar='N',
                        help='maximal number of main iterations (default:500)')
    # 输出相关
    parser.add_argument('--model_dir', help='dir to save trained model',
                        default="/tmp/tbase/models", type=str)
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='interval between training status(default:10)')
    parser.add_argument('--save-model-interval', type=int, default=5)
    parser.add_argument('--tensorboard_dir', default='/tmp/tbase/tensorboard',
                        type=str,
                        help='Directory to save learning curve data.')
    parser.add_argument('--log-action', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    # 运行参数
    parser.add_argument('--play', default=False, action='store_true')
    return parser.parse_args()
