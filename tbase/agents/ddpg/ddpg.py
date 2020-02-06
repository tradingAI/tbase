import torch

from tbase.common.torch_utils import soft_update


# step for batch data
def step(policy_net, value_net, optimizer_policy, optimizer_value, states,
         actions, rewards, states_next, dones, gamma, tau,
         target_policy_net, target_value_net):
    """update critic"""
    # target_q
    target_act_next = policy_net(states_next)
    target_q_next = value_net(states, target_act_next)
    target_q = rewards + torch.mul(target_q_next, (dones * gamma))

    q = value_net(states, actions)
    value_loss = torch.nn.MSELoss()(q, target_q)
    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()

    """update policy"""
    reg_values, actions_new = policy_net(states, reg_value=True)
    values = value_net(states, actions_new)
    loss_reg = torch.mean(torch.pow(reg_values, 2)) * 1e-2
    policy_loss = loss_reg + values
    optimizer_policy.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 40)
    optimizer_policy.step()

    soft_update(target_policy_net, policy_net, tau)
    soft_update(target_value_net, value_net, tau)
