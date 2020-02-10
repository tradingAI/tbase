# tbase

Baselines of trade agents use reinforcement learning algorithms(PyTorch).

支持环境： python3(>=3.5)

# 安装

设置 tushare token[(token注册链接:https://tushare.pro/register?reg=124861)](https://tushare.pro/register?reg=124861):

```
export TUSHARE_TOKEN=YOUR_TOKEN
```

[docker install](https://docs.docker.com/install/)

# Features(开发中, 欢迎一起完善)

- [ ] 加速

  - [x] 多进程CPU并行: 多进程运行独立的Enviroment进行探索
  - [ ] 多进程单GPU并行
  - [ ] 多进程多GPU并行

- [ ] 支持单Agent与多Agent, 例如: MADDPG

- [x] 通过运行参数选择:

  - [x] 环境
  - [x] 算法
  - [x] Policy-Net
  - [x] Value-Net

- 支持RL算法:

  - [ ] One Agent

    - [x] DDPG
    - [ ] Dueling DDPG
    - [ ] A2C
    - [ ] PPO
    - [ ] PPO2
    - [ ] ACKTR
    - [ ] GAIL
    - [ ] TD3

  - [ ] Multi-Agent

    - [ ] MADDPG

- 自定义Net

  - [x] LSTM-MLP
  - [x] LSTM_Merge_MLP
  - [ ] MLP
  - [ ] LSTM
  - [ ] CNN

  - [ ] CNN-MLP

# 训练与保存模型

# 模型评估

- [ ] 绝对收益率(Absolute Return)
- [ ] 额外收益率(Excess Return)

  - [ ] 相对于"买入持有"策略
  - [ ] 相对于基线策略比如"上证300"

- [ ] 最大回撤: 在选定周期内任一历史时点往后推，产品净值走到最低点时的收益率回撤幅度的最大值

- [ ] 夏普比率: 投资组合每承受一单位总风险，会产生多少的超额报酬

# 加载模型
