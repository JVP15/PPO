import numpy as np
# import torch
# import torch.nn as nn
import tensorflow as tf
from tensorflow import clip_by_value

# class NormalModule(nn.Module):
#     def __init__(self, inp, out, activation=nn.Tanh):
#         super().__init__()
#         self.m = nn.Linear(inp, out)
#         log_std = -0.5 * np.ones(out, dtype=np.float32)
#         self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
#         self.act1 = activation

#     def forward(self, inputs):
#         mout = self.m(inputs)
#         vout = torch.exp(self.log_std)
#         return mout, vout

def reward_to_go(trajectory, time, gamma):
    """
    This calculates the reward to go for the given trajectory (a list of (s, a, r) tuples) starting at the given time
    Equivalent to R_to_go = sum t=t` to T (gamma^(t-t`) * r_t)
    :param trajectory: A list of (s, a, r) tuples
    :param time: The time to start calculating the reward from
    :param gamma: The discount factor
    :return: The reward to go
    """
    reward = 0

    # loop over the trajectory from the given time until the end
    for count, timestep in enumerate(trajectory[time:]):
        #print(timestep[0], '\n', timestep[1],'\n', timestep[2], '\n')
        reward += timestep[2] * (gamma ** count)

    #print('reward to go =', reward)
    return reward

# loss with clipping


def policy_gradient_loss(trajectories, policy_func, value_func, gamma):
    """
    Computes the loss for the policy gradient algorithm with reward-to-go instead of an advantage function.
    Loss = E[log(pi(a|s)) * (R_to_go)]

    :param trajectories: A list of trajectories, where each element of each trajectory is (s, a, r)
    :param policy_func: The policy function pi(a|s).
    :param value_func: The value function V(s) (not actually used in this function, but here to match the signature)
    :param gamma: The discount factor.
    :return: The loss for the policy gradient algorithm.
    """

    loss = 0

    # loop over all trajectories in the batch
    for trajectory in trajectories:
        # loop over all timesteps in the trajectory and compute the loss for each timestep
        #print('trajectory =', trajectory)
        for time, timestep in enumerate(trajectory):
            # calculate the reward to go for this timestep
            reward = reward_to_go(trajectory, time, gamma)

            # calculate the loss for this timestep
            loss += tf.math.log(policy_func(timestep[1], timestep[0])) * reward

    # average the loss over all trajectories
    loss *= 1 / len(trajectories)  * 1 / len(trajectories[0])

    return loss

def surrogate_loss(trajectories, policy_func, value_func, gamma, clip, ratio):
    loss = 0
    for trajectory in trajectories:
        for time, timestep in enumerate(trajectory):
            loss += ratio(timestep[0]) * advantage_function(trajectory, value_func, gamma, time)

    loss *= 1 / len(trajectories) * 1 / len(trajectories[0])

    return loss

def surrogate_loss_clipped(trajectories, policy_func, value_func, gamma, clip, ratio_func):
    """
    Computes the loss for the policy gradient algorithm with the generalized advantage function and clipping.
    Loss = min(ratio * A(s, a), clip(ratio) * A(s, a))

    :param trajectories: A list of trajectories, where each element of each trajectory is (s, a, r)
    :param clip: The clipping parameter.
    :param ratio_func: The ratio function function for the current and old policies.
    :param value_func: The value function V(s))
    :return: The loss for the policy gradient with clipping algorithm.
    """
    loss = 0
    for trajectory in trajectories:
        for time, timestep in enumerate(trajectory):

            ratio = ratio_func(timestep[0])
            A = advantage_function(trajectory, value_func, gamma, time)
            loss1= clip(ratio, A) * A
            loss2= ratio * A
            loss += tf.minimum(loss1,loss2)

    loss *= 1 / len(trajectories) * 1 / len(trajectories[0])
    return loss

def clip(x, advantage, epsilon = .2):
    # the value is clipped at 1-epsilon if the advantage is negative and 1+epsilon if the advantage is positive
    if advantage > 0:
        return tf.clip_by_value(x, x, 1 + epsilon)
    else:
        return tf.clip_by_value(x, 1 - epsilon, x)

def delta(r, v, gamma, time):
    return r + gamma * v[time + 1] - v[time]

def advantage_function(trajectory, value_func, gamma, time):
    advantage = 0
    # pre-calculate the value function for each state in the timestep so that we don't have to do it multiple times
    values = np.array([value_func(timestep[0]) for timestep in trajectory], dtype=np.float32)

    for count, timestep in enumerate(trajectory[time:-1]):
        reward = timestep[2]
        advantage += delta(reward, values, gamma, count)

    return advantage