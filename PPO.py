import gym
import numpy as np
from numpy.random import normal
import tensorflow as tf


class PPOAgent(object):
    def __init__(self, input_size, output_size, log_std):
        self.input_size = input_size
        self.output_size = output_size
        self._log_std = log_std * np.ones(output_size)

        # TODO: implement Mu as an MLP that takes state as input and outputs action mean
        self._mu = np.zeros(output_size)

    def train(self):
        pass

    def mu(self, state):
        return self._mu

    def std(self, state = None):
        return np.exp(self._log_std)

    def action(self, state):
        mu = self.mu(state)
        std = self.std(state)

        # Choose action from a normal distribution using mu and std (reparametarized to work with derivatives)
        action = mu + std * normal(0, 1, size = self.output_size)

        return action
