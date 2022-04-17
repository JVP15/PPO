import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf

class NormalModule(nn.Module):
    def __init__(self, inp, out, activation=nn.Tanh):
        super().__init__()
        self.m = nn.Linear(inp, out)
        log_std = -0.5 * np.ones(out, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.act1 = activation

    def forward(self, inputs):
        mout = self.m(inputs)
        vout = torch.exp(self.log_std)
        return mout, vout


# this is the loss function for policy gradient
# use @tf.function to make this a tensorflow function so that it can be differentiated by tf.GradientTape
@tf.function
def policy_gradient_loss(trajectories, policy_func, value_func, gamma=0.99):
    """
    Computes the loss for the policy gradient algorithm.
    :param trajectories: A list of trajectories, where each element of each trajectory is (s, a, r)
    :param policy: The policy function pi(a|s).
    :param value: The value function V(s).
    :param gamma: The discount factor.
    :return: The loss for the policy gradient algorithm.
    """

    pass

