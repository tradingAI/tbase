# tbase

Baselines of reinforcement learning trading agents(PyTorch based).

支持环境： Python 3.5–3.7

# 安装

设置 tushare token[(token注册链接:https://tushare.pro/register?reg=124861)](https://tushare.pro/register?reg=124861):

```
export TUSHARE_TOKEN=YOUR_TOKEN
```

**1\. Mac OSX/Ubuntu**

- 依赖: `python3, pip`, 如果没有安装可参考[Ubuntu 18.04 Dockerfile](Dockerfile)
- [tgym](https://github.com/iminders/tgym)
  ```
  git clone https://github.com/iminders/tgym
  cd tgym
  pip install -r requirements.txt
  pip install -e .
  ```
- tbase

  ```
  git clone https://github.com/iminders/tbase
  cd tbase
  pip install -r requirements.txt
  pip install -e .
  ```

**2\. Docker**

- 1. [docker install](https://docs.docker.com/install/)
- 2. `docker run -it aiminders/trade:tbase bash` (如果下载时间过长，可以选执行第iii步，再执行第ii步)
- 3. (可选)Build your docker image， 参考 [build-docker-image.sh](build-docker-image.sh)


# Features(In progress)

- [ ] 加速

  - [x] 多进程CPU并行: 多进程运行独立的Enviroment进行探索
  - [x] 多进程单GPU并行
  - [ ] 多进程多GPU并行
  - [ ] 混合精度训练(Apex)

- [x] 通过运行参数选择:

  - [x] 环境
  - [x] 算法
  - [x] Policy-Net
  - [x] Value-Net

- 支持RL算法:

  - [ ] 单Agent

    - [x] DDPG
    - [ ] TD3(Twin Delayed Deep Deterministic Policy Gradients)
    - [ ] A2C
    - [ ] PPO
    - [ ] PPO2
    - [ ] ACKTR
    - [ ] GAIL

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
python3 -m tbase.run --alg ddpg --num_env 4 --gamma 0.53 --seed 9 --print_action
```

默认参数:
- scenario: "average", 平均分仓操作
- codes: "000001.SZ", 平安银行
- indexs: "000001.SH,399001.SZ", [000001.SH:沪指, 399001.SZ: 深指](https://tushare.pro/document/2?doc_id=94)
- start: "20190101", 训练开始时间
- end: "201901231", 训练结束时间
- max_iter_num: "500", 训练轮数
- num_env: "2", 并行进程数
- seed: "None", 系统随机种子
- print_action: "False", 随机打印action的值，方便查看action分布状况
- reward_fn: "daily_return_with_chl_penalty", env reward function
- run_id: "1", 运行序号, 方便查看相同参数多次运行结果差异
- [其他参数](tbase/common/cmd_util.py)

**Defalut policy net setting(actor)**

![actor](tbase/agents/ddpg/images/policy.png)

**Defalut value net setting(critic)**

![critic](tbase/agents/ddpg/images/value.png)

运行tensorboard

`tensorboard --logdir=/tmp/tbase/tensorboard`

可以在[http://localhost:6006](http://localhost:6006/)中查看训练的loss, reward ,portfolio, time等指标

![loss](images/default_param.png)

# 加载模型

# 评估

- [x] 训练周期内的评估指标
- [ ] 模型随未来时间段的性能指标
- [ ] 滑动窗口更新模型, 在评估周期内，每隔一个窗口T，重新训练一次模型，当T>评估周期时，等价于固定模型

## 评估指标

- [x] 绝对收益率(Absolute Return)
- [ ] 额外收益率(Excess Return)

  - [ ] 相对于"买入持有"策略
  - [ ] 相对于基线策略比如"上证300"

- [ ] 最大回撤: 在选定周期内任一历史时点往后推，产品净值走到最低点时的收益率回撤幅度的最大值

- [ ] 夏普比率: 投资组合每承受一单位总风险，会产生多少的超额报酬

## Contribution
- Fork this repo
- Add or change code && **Please add tests for changes**
- Test
  - step1. 设置[docker-compose](docker-compose.yml)需要的环境变量: BAZEL_USER_ROOT, OUTPUT_DIR, TUSHARE_TOKEN
  - step2. `docker-compose up`
- Send pull request

# 如何增加agent
1. Fork  https://github.com/iminders/tbase
2. 在tbase.agents下添加目录, 例如: ddpg
3. 新建agent.py, 在类名为Agent的类中实现你的agent(继承tbase.common.base_agent.BaseAgent)
4. **添加单元测试**
  - step1. 设置[docker-compose](docker-compose.yml)需要的环境变量: BAZEL_USER_ROOT, OUTPUT_DIR, TUSHARE_TOKEN
  - step2. `docker-compose up`
5. 发起pull request

# 待优化

- [x] [bazel build](https://bazel.build/)
- [x] 版本管理
- [ ] unittest 补充
- [ ] baseline模型共享(百度网盘)
- [ ] 由于计算资源有限，为所有的算法跑完A股中所有的股票，需要花费大量的时间，希望有空闲计算资源的朋友，可以跑一下模型，更新到repo中, 以方便其他人复现, 包含以下信息
  - run.sh(运行脚本)
  - 参数设置
  - performance: 5次平均 best portfolio
  - 百度云盘链接

线上交流方式

- QQ群: 477860214
