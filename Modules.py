import numpy as np
import tensorflow as tf
from tensorflow import clip_by_value


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
        r = timestep[2]
        reward += r * (gamma ** count)

    #print('reward to go =', reward)
    return reward

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
    values = value_func(tf.convert_to_tensor([value_func(timestep[0]) for timestep in trajectory]), batch=True)

    for count, timestep in enumerate(trajectory[time:-1]):
        reward = timestep[2]
        advantage += delta(reward, values, gamma, count)

    return advantage

def policy_gradient_loss(trajectories, policy_func, gamma, **kwargs):
    """
    Computes the loss for the policy gradient algorithm with reward-to-go instead of an advantage function.
    Loss = E[log(pi(a|s)) * (R_to_go)]

    :param trajectories: A list of trajectories, where each element of each trajectory is (s, a, r)
    :param policy_func: The policy function pi(a|s).
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
            state = timestep[0]
            action = timestep[1]
            # calculate the loss for this timestep
            loss += tf.math.log(policy_func(action, state)) * reward

    # normalize the loss
    loss *= 1 / len(trajectories) * 1 / len(trajectories[0])

    return loss

def policy_gradient_loss_advantage(trajectories, policy_func, value_func, gamma, **kwargs):
    """
    Computes the loss for the policy gradient algorithm with the advantage function.
    Loss = E[log(pi(a|s)) * A_t]

    :param trajectories: A list of trajectories, where each element of each trajectory is (s, a, r)
    :param policy_func: The policy function pi(a|s).
    :param value_func: The value function V(s)
    :param gamma: The discount factor.
    :return: The loss for the policy gradient algorithm.
    """

    loss = 0

    # loop over all trajectories in the batch
    for trajectory in trajectories:
        # loop over all timesteps in the trajectory and compute the loss for each timestep
        # print('trajectory =', trajectory)
        for time, timestep in enumerate(trajectory):
            # calculate the reward to go for this timestep
            advantage = advantage_function(trajectory, value_func, time, gamma)
            state = timestep[0]
            action = timestep[1]
            # calculate the loss for this timestep
            loss += tf.math.log(policy_func(action, state)) * advantage

    # normalize the loss
    loss *= 1 / len(trajectories) * 1 / len(trajectories[0])

    return loss

def surrogate_loss(trajectories, ratio_func, value_func, gamma, **kwargs):
    """
    Computes the loss for the surrogate policy gradient algorithm.
    Loss = E[ratio * A_t]
    :param trajectories: A list of trajectories, where each element of each trajectory is (s, a, r)
    :param ratio_func: The ratio function r(a|s) = pi(a|s) / pi_old(a|s)
    :param value_func: The value function V(s)
    :param gamma: The discount factor.
    :return: The loss for the policy gradient algorithm.
    """
    loss = 0
    for trajectory in trajectories:
        for time, timestep in enumerate(trajectory):
            state = timestep[0]
            action = timestep[1]
            loss += ratio_func(action, state) * advantage_function(trajectory, value_func, gamma, time)

    # normalize the loss
    loss *= 1 / len(trajectories) * 1 / len(trajectories[0])

    return loss

def surrogate_loss_clipped(trajectories,value_func, gamma, clip_value, ratio_func, **kwargs):
    """
    Computes the loss for the policy gradient algorithm with the generalized advantage function and clipping.
    Loss = min(ratio * A(s, a), clip(ratio) * A(s, a))

    :param trajectories: A list of trajectories, where each element of each trajectory is (s, a, r)
    :param clip_value: The clipping parameter.
    :param ratio_func: The ratio function r(a|s) = pi(a|s) / pi_old(a|s)
    :param value_func: The value function V(s)
    :return: The loss for the policy gradient with clipping algorithm.
    """
    loss = 0
    for trajectory in trajectories:
        for time, timestep in enumerate(trajectory):
            state = timestep[0]
            action = timestep[1]

            ratio = ratio_func(action, state)
            A = advantage_function(trajectory, value_func, gamma, time)
            loss1= clip(ratio, A, clip_value) * A
            loss2= ratio * A
            loss += tf.minimum(loss1,loss2)

    # normalize the loss
    loss *= 1 / len(trajectories) * 1 / len(trajectories[0])

    return loss

def value_loss(trajectories, value_func, gamma):
    """
    Computes the mean square error loss for the value function.
    Loss = E[(V(s) - R_to_go)^2]

    :param trajectories: A list of trajectories, where each element of each trajectory is (s, a, r)
    :param value_func: The value function V(s)
    :param gamma: The discount factor.
    :return: The loss for the value function.
    """
    mse = tf.keras.losses.MeanSquaredError()

    v_true = []
    v_pred = []

    # loop over all trajectories in the batch
    for trajectory in trajectories:
        # loop over all timesteps in the trajectory and compute the reward-to-go for each timestep
        for time, timestep in enumerate(trajectory):
            # calculate the reward to go for this timestep
            v_true.append(reward_to_go(trajectory, time, gamma))

            # calculate the value function for this timestep
            v_pred.append(value_func(timestep[0]))

    return mse(v_true, v_pred)

