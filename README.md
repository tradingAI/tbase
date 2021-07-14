![Test](https://github.com/tradingAI/tbase/workflows/Test/badge.svg?branch=master)
[![PyPI version](https://badge.fury.io/py/tbase.svg)](https://badge.fury.io/py/tbase)
[![Coverage Status](https://coveralls.io/repos/github/tradingAI/tbase/badge.svg?branch=dev)](https://coveralls.io/github/tradingAI/tbase?branch=dev)
# tbase

基于强化学习的交易算法Baselines(Pytorch&Tensorflow2实现). 旨在提供一个在中国股市上可复现的算法Baselines.

支持环境： Python 3.5–3.7

# 安装

设置 tushare token[(Tushare token注册申请)](https://tushare.pro/register?reg=124861):

```
export TUSHARE_TOKEN=YOUR_TOKEN

pip install tbase --upgrade
```
# Example
```
python3 -m tbase.run --alg ddpg
```


# Features(In progress)
- [x] 可以完美复现训练过程, [How?](docs/reproducible.md)
- [ ] 加速
  - [x] 多进程CPU并行: 多进程运行独立的Enviroment进行探索
  - [x] 多进程单GPU并行
  - [ ] 多进程多GPU并行
  - [ ] 分布式多机多卡并行: 参考 [menger](https://ai.googleblog.com/2020/10/massively-large-scale-distributed.html)

- 支持算法:

  - [ ] 单Agent
    - [ ] (Double Dueling)DQN
    - [x] DDPG
    - [x] TD3(Twin Delayed Deep Deterministic Policy Gradients)
    - [ ] A2C-ACKTR
      - [x] A2C
      - [ ] ACKTR
    - [ ] PPO
    - [ ] PPO2
    - [ ] GAIL
    - [ ] OPD

  - [ ] 多Agent
    - [ ] MADDPG

- 自定义Net

  - [x] LSTM-MLP
  - [x] LSTM_Merge_MLP
  - [ ] MLP
  - [ ] LSTM
  - [ ] CNN
  - [ ] CNN-MLP

# 训练

例如 ddpg

```
python3 -m tbase.run --alg ddpg --num_env 1 --gamma 0.53 --seed 9 --print_action
```

默认参数:
- scenario: "average", 平均分仓操作
- codes: "000001.SZ", 平安银行
- indexs: "000001.SH,399001.SZ", [000001.SH:沪指, 399001.SZ: 深指](https://tushare.pro/document/2?doc_id=94)
- start: "20190101", 训练开始时间
- end: "201901231", 训练结束时间
- max_iter_num: "500", 训练轮数
- num_env: "1", 并行进程数, 当`num_env=1`时，不使用多进程方式运行
- seed: "None", 系统随机种子
- print_action: "False", 随机打印action的值，方便查看action分布状况
- reward_fn: "daily_return_with_chl_penalty", env reward function
- run_id: "1", 运行序号, 方便查看相同参数多次运行结果差异
- debug: "False", debug模式，打印具体的交易信息
- eval: "False", 回测评估，为True时，不训练，只加载模型进行回测
- [其他参数](tbase/common/cmd_util.py)


**Defalut policy net setting(actor)**

![actor](tbase/agents/ddpg/images/policy.png)

**Defalut value net setting(critic)**

![critic](tbase/agents/ddpg/images/value.png)

运行tensorboard

`tensorboard --logdir=/tmp/tbase/tensorboard`

可以在[http://localhost:6006](http://localhost:6006/)中查看训练的loss, reward ,portfolio, time等指标

![loss](images/default_param.png)

# 评估&&回测

- [x] 训练周期内的评估指标
- [x] 回测: 模型在训练周期外(评估周期)的评估指标
  ```
  # lookback=10， 若测试从20200101开始, 则eval_start往前推10个交易日
  python -m tbase.run --alg ddpg --eval --eval_start 20191218 --eval_end 20200223 --seed 9
  ```
- [ ] 滑动窗口更新模型, 在评估周期内，每隔一个窗口T，重新训练一次模型，当T>评估周期时，等价于固定模型

## 评估指标

- [x] 绝对收益率(Absolute Return)
- [ ] 额外收益率(Excess Return)

  - [x] 相对于"买入持有"策略
  - [ ] 相对于基线策略比如"沪深300"

- [x] 最大回撤: 在选定周期内任一历史时点往后推，净值走到最低点时的收益率回撤幅度的最大值

- [x] 夏普比率: 投资组合每承受一单位总风险，会产生多少的超额报酬

- [x] 年化收益率

# 待优化

- [ ] 考虑到部署到线上的方便性, 更换tensorflow实现
- [ ] 参数配置管理: yml
- [ ] 使用cpprb作业replay buffer
- [ ] 测试: 提升 pytest 覆盖率([coveralls.io](https://coveralls.io/))
- [ ] 模型分享，评估，部署(wandb)
  - 模型榜单排名
  - performance: 参考评估指标(去除绝对收益率)
- [ ] 大规模训练: 参考 [menger](https://ai.googleblog.com/2020/10/massively-large-scale-distributed.html)
  - [ ] 分布式, evnviroment explorer, 方便无限scale
  - [ ] 多个buffer
    - [ ] sample
    - [ ] model update & inference
  - [ ] 监控: explorer, buffer, learner, 吞吐量，以方便定位bottle neck
  - [ ] stock2vector

**由于计算资源有限，为所有的算法跑完A股中所有的股票，需要花费大量的时间，希望有空闲计算资源的朋友，可以跑一下模型，将实验结果分享出来**

线上交流方式

- QQ群: 477860214
