# 如何完美复现训练过程

Note(wen):
- 多进程方式运行不能保证完美复现, [Why?](https://github.com/iminders/tbase/issues/2)
- 分别GPU和CPU上运行的结果并不一致, [PyTorch Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)
- 在相同类型设备(GPU/CPU)上运行, 完美复现

设置“非多进程”的方式训练: ```--num_env 1``

以`ddpg`为例:
```
python -m tbase.run --alg ddpg --num_env 1
```
