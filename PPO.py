import gym
import numpy as np
from numpy.random import normal
import tensorflow as tf
import tensorflow.keras.layers.Dense as Dense
from Modules import policy_gradient_loss

class PPOAgent(object):
    def __init__(self, input_size, output_size,
                 log_std=.01,
                 num_trajectories=32,
                 trajectory_length=100,
                 n_epochs=10,
                 learning_rate=3e-4,
                 gamma=.99,
                 loss_function=policy_gradient_loss):

        self.input_size = input_size
        self.output_size = output_size
        self._log_std = log_std * np.ones(output_size)
        self._num_trajectories = num_trajectories
        self._trajectory_length = trajectory_length
        self._n_epochs = n_epochs
        self._learning_rate = learning_rate
        self._gamma = gamma
        self._loss_function = loss_function

        # mu is an MLP with one hidden layer that takes the state as input and outputs the mean of the action
        self._mu = tf.keras.Sequential([Dense(32, input_shape=(input_size,), activation='tanh'),
                                        Dense(32, activation='tanh'),
                                        Dense(output_size, activation='tanh')])

    def train(self, max_iterations, env):
        # loop over iterations
        for iteration in range(max_iterations):
            # this is the 'replay buffer' from the README. It is a collection of trajectories. Each trajectory
            #   is a list of (state_t, action_t, reward_t) tuples where t is the time step.
            # e.g., trajectories[0][2] will return the state, action, and reward at time step 2 in trajectory 0.
            trajectories = self._collect_trajectories(env)

            # gradient tape allows us to perform automatic differentiation
            with tf.GradientTape(persistent=True) as tape:
                loss = self._loss_function(trajectories, policy_func=self.pi, value_func=None, gamma=self._gamma)

    # use @tf.function to make this a tensorflow function so that it can be differentiated by tf.GradientTape
    @tf.function
    def mu(self, state):
        return self._mu(state)

    def std(self, state = None):
        return np.exp(self._log_std)

    # use @tf.function to make this a tensorflow function so that it can be differentiated by tf.GradientTape
    @tf.function
    def pi(self, state, action):
        """This is the policy pi(a|s), which computes the probability that the agent will take action a given the state s.
        """
        # since we sample from a gaussian distribution to get the action, we can use the gauusian pdf to compute the
        #   probability of the action.

        mu = self.mu(state)
        std = self.std(state)

        # compute the probability of the action given the state
        pi = tf.exp(-(action - mu)**2 / (2 * std**2)) / (2 * np.pi * std**2)

        return pi

    def action(self, state):
        mu = self.mu(state)
        std = self.std(state)

        # Choose action from a normal distribution using mu and std (reparametarized to work with derivatives)
        action = mu + std * normal(0, 1, size = self.output_size)

        return action

    def _collect_trajectories(self, env):
        trajectories = []

        # collect all trajectories for the current iteration
        for trajectory_number in range(self._num_trajectories):
            # collect data from a single trajectory/episode
            trajectory = []

            # reset the environment
            state = env.reset()

            # collect the state, action, and reward of each time step for trajectory_length time steps
            for time_step in range(self._trajectory_length):
                # get the action from the agent
                action = self.action(state)

                # take a step in the environment
                next_state, reward, done, _ = env.step(action)
                trajectory.append((state, action, reward))
                state = next_state

            trajectories.append(trajectory)

        return trajectories
