# 如何完美复现训练过程

Note: 多进程方式运行不能保证完美复现, [Why?](https://github.com/iminders/tbase/issues/2)

设置不用多进程的方式训练: ```--num_env 1``

以`ddpg`为例:
```
python -m tbase.run --alg ddpg --num_env 1
```
