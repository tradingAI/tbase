# A2C
A2C is a synchronous, deterministic variant of Asynchronous Advantage Actor Critic (A3C)
- Original paper:
    - [A3C](https://arxiv.org/pdf/1602.01783.pdf)
    - [Acktr:(ACKTR (pronounced “actor”) — Actor Critic using Kronecker-factored Trust Region)](https://arxiv.org/pdf/1708.05144.pdf)
- Baselines post: TODO
- `python3 -m tbase.run --alg=a2c_acktr --scenario=average --max_iter_num=500 --policy_net=LSTM_MLP_A2C --seed 9` runs the algorithm for about 122000 days = 500 episode on a tgym.average environment. See help (`-h`) for more options.


## A3C

Note | meaning
:-- |:--
$$\varepsilon$$ | environment
$$t$$ | time step
$$s_t$$ | state at step $$t$$
$$a_t$$ | action accoding its policy
$$\pi$$ | policy
$$r_t$$ | reward at step $$t$$
$$\gamma$$ | discount factor
$$R_t$$ | Return from time step t with discount factor $$\gamma$$
$$Q^{\pi}(s, a) = E[R_t | s_t=s, a]$$ | expected return for select action $$a$$ in state $$s$$ and following policy $$\pi$$
$$Q^*(s, a)=max_{\pi}Q^{\pi}(s, a)$$ | optimal value function gives the maximum action value for state $$s$$ and action $$a$$ achievable by any policy
$$V^{\pi} = E[R_t | S_t = s]$$ | the expected return for following policy $$\pi$$ from state $$s$$
$$Q(s, a; \theta$$)| approximate action-value function with parameters $$\theta$$
$$s'$$ | next state
$$a'$$ | next action
$$b_t(s_T)\approx V^{\pi}(s_t)$$ | A learned estimate of the value function's baseline
$$A(a_t, s_t)=Q(a_t, s_t) - V(s_t)$$ | estimate of the $$advantage$$ of action $$a_t$$ in state $$s_t$$

## 异步RL框架
- 单机多线程的方式，降低了gradients和parameters通信成本
- Hogwild! style updates for training
- 并行运行多个actor, 探索环境的不同部分
- 在多个环境中的不同部分并行探索，online updates, 降低actor与环境之间的相关性
- 不使用replay memory
优势：
- 减少训练时间，与并行的actor数量呈线性关系
- on-policy 的方式

Asynchronous advantage actor-critic(A3C)
  - policy: $$\pi(a_t|s_t;\theta)$$
  - estimate value function: $$V(s_t; \theta)$$
  - 每$$t_{max}$$步或者terminal state到达更新policy&value function
  - Optimization: RMSProp with shared statistics 更优
![a3c](images/a3c.png)
