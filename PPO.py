import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import clone_model
from Modules import policy_gradient_loss, surrogate_loss_clipped, surrogate_loss


class PPOAgent(object):
    def __init__(self, input_size, output_size,
                 log_std=.01,
                 num_trajectories=32,
                 trajectory_length=100,
                 n_epochs=10,
                 learning_rate=3e-4,
                 gamma=.99,
                 clip_value = 0.2,
                 loss_function=surrogate_loss):
        # clip value
        self._clip_value = clip_value
        self.input_size = input_size
        self.output_size = output_size
        self._log_std = log_std * np.ones(output_size)
        self._num_trajectories = num_trajectories
        self._trajectory_length = trajectory_length
        self._n_epochs = n_epochs
        self._learning_rate = learning_rate
        self._gamma = gamma
        self._loss_function = loss_function
    
        # mu is an MLP takes the state as input and outputs the mean of the action
        self._mu = tf.keras.Sequential([Dense(32, input_shape=(input_size,), activation='tanh'),
                                        Dense(32, activation='tanh'),
                                        Dense(output_size, activation='tanh')])

        self._mu_old = tf.keras.Sequential([Dense(32, input_shape=(input_size,), activation='tanh'),
                                        Dense(32, activation='tanh'),
                                        Dense(output_size, activation='tanh')])

        # this is the value function for the policy
        self._value = tf.keras.Sequential([Dense(32, input_shape=(input_size,), activation='tanh'),
                                        Dense(32, activation='tanh'),
                                        Dense(output_size, activation='tanh')])

    def train(self, env, max_iterations, log_interval = 200, eval_interval = 200, eval_episodes=5):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self._learning_rate)

        # loop over iterations
        for iteration in range(max_iterations):
            # this is the 'replay buffer' from the README. It is a collection of trajectories. Each trajectory
            #   is a list of (state_t, action_t, reward_t) tuples where t is the time step.
            # e.g., trajectories[0][2] will return the state, action, and reward at time step 2 in trajectory 0.
            trajectories = self._collect_trajectories(env)

            # gradient tape allows us to perform automatic differentiation
            with tf.GradientTape(persistent=True) as tape:
                
                # uncomment this for loss function using vanilla policy gradient, the only difference is the parameters passed.
                # loss = -self._loss_function(trajectories, policy_func=self.pi, value_func=self.V, gamma=self._gamma)  
                loss = -self._loss_function(trajectories, policy_func=self.pi, value_func=self.V, gamma=self._gamma, clip = self._clip_value, ratio = self.ratio)

            mu_weights = self._mu.get_weights()
            # compute the gradients of the loss with respect to the policy and value parameters
            # if you aren't using the value function, then we ignore the gradient of the value function using the 'unconnected_gradients' argument
            gradients = tape.gradient(loss, self._mu.trainable_variables + self._value.trainable_variables,
                                      unconnected_gradients=tf.UnconnectedGradients.ZERO)
            
            self._mu_old.set_weights(mu_weights)
            
            optimizer.apply_gradients(zip(gradients, self._mu.trainable_variables + self._value.trainable_variables))

            if iteration % log_interval == 0:
                print(f'iteration: {iteration}, loss: {loss}')

            if iteration % eval_interval == 0:
                mean_reward = self.evaluate(env, eval_episodes)
                print(f'iteration: {iteration}, mean reward over {eval_episodes} episodes: {mean_reward}')

    def mu(self, state):
        # tensorflow models expect inputs to be in the form of a batch of examples, so we have to add a batch dimension before calling the model
        state = tf.expand_dims(state, axis=0)
        return self._mu(state)

    def mu_old(self, state):
        # tensorflow models expect inputs to be in the form of a batch of examples, so we have to add a batch dimension before calling the model
        state = tf.expand_dims(state, axis=0)
        return self._mu_old(state)

    def V(self, state):
        # tensorflow models expect inputs to be in the form of a batch of examples, so we have to add a batch dimension before calling the model
        state = tf.expand_dims(state, axis=0)
        return self._value(state)

    def std(self, state = None):
        # In our original implementation, std is a tunable hyperparameter, but we may make it an MLP based on state, so
        #  I've added this function to future-proof the code.
        return np.exp(self._log_std)

    def ratio(self, state):
        mu_old = self.mu_old(state)
        mu = self.mu(state)
        return mu / mu_old
    
    def pi(self, action, state):
        """This is the policy pi(a|s), which computes the probability that the agent will take action a given the state s.
        """
        # since we sample from a gaussian distribution to get the action, we can use the gauusian pdf to compute the
        #   probability of the action.

        mu = self.mu(state)
        std = self.std(state)

        # compute the probability of the action given the state. Probability density function taken from wolfram alpha:
        # https://reference.wolfram.com/language/ref/NormalDistribution.html
        prob = 1 / (std * tf.sqrt(2 * np.pi)) * tf.exp(-(action - mu)**2 / (2 * std**2))
        return prob

    def action(self, state):
        mu = self.mu(state)
        std = self.std(state)

        # Choose action from a normal distribution using mu and std (reparametarized to work with derivatives)
        action = mu + std * np.random.normal(0, 1, size = self.output_size)
        return action

    def _collect_trajectories(self, env):
        trajectories = []

        # collect all trajectories for the current iteration
        for _ in range(self._num_trajectories):
            # collect data from a single trajectory/episode
            trajectory = []

            # reset the environment
            state = env.reset()

            # collect the state, action, and reward of each time step for trajectory_length time steps
            for _ in range(self._trajectory_length):
                # get the action from the agent
                action = self.action(state)

                # take a step in the environment
                next_state, reward, done, _ = env.step(action)
                trajectory.append((state, action, reward))
                state = next_state

            trajectories.append(trajectory)

        return trajectories

    def evaluate(self, env, num_episodes):
        """Evaluate the agent's performance on the environment. Returns the average return of the agent over the
        specified number of episodes.
        """
        total_reward = 0.0

        for _ in range(num_episodes):
            # reset the environment
            state = env.reset()

            while True:
                # get the action from the agent
                action = self.action(state)

                # take a step in the environment
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break

        return total_reward / num_episodes