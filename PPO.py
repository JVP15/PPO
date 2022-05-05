import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import clone_model
from Modules import policy_gradient_loss, value_loss, reward_to_go
import matplotlib.pyplot as plt


class PPOAgent(object):
    def __init__(self, input_size, output_size,
                 log_std=.01,
                 num_trajectories=32,
                 trajectory_length=100,
                 n_epochs=10,
                 learning_rate=3e-4,
                 gamma=.99,
                 clip_value = 0.2,
                 loss_function=policy_gradient_loss):
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
        self._actor_loss_function = loss_function
        self._value_loss_function = value_loss

        self.rewards = [] # this tracks the reward-to-go for each iteration
        self.losses = [] # this tracks the loss for each iteration
    
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

            # for speed's sake, only collect the reward-to-go for the first trajectory in the iteration
            self.rewards.append(reward_to_go(trajectories[0], time=0, gamma=self._gamma))

            # gradient tape allows us to perform automatic differentiation
            with tf.GradientTape(persistent=True) as tape:

                actor_loss = self._actor_loss_function(trajectories, policy_func=self.pi, value_func=self.V, gamma=self._gamma, clip_value=self._clip_value, ratio_func = self.ratio)
                value_loss = self._value_loss_function(trajectories, value_func=self.V, gamma=self._gamma)

                # for some reason, we need to negate the total loss for TF's ADAM to work properly
                total_loss = -(actor_loss - value_loss)

            self.losses.append(-total_loss.numpy())

            mu_weights = self._mu.get_weights()
            # compute the gradients of the loss with respect to the policy and value parameters
            # if you aren't using the value function, then we ignore the gradient of the value function using the 'unconnected_gradients' argument
            mu_gradients = tape.gradient(total_loss, self._mu.trainable_variables)
            value_gradients = tape.gradient(total_loss, self._value.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)

            # save the current mu network before we update it with the gradients
            self._mu_old.set_weights(mu_weights)
            
            optimizer.apply_gradients(zip(mu_gradients, self._mu.trainable_variables))
            optimizer.apply_gradients(zip(value_gradients, self._value.trainable_variables))

            if iteration % log_interval == 0:
                print(f'iteration: {iteration}, loss: {total_loss}')

                # sample 16x16 evenly distributed points in the state space to visualize the value function
                theta = np.linspace(-np.pi, np.pi, 16)
                theta_dot = np.linspace(-8, 8, 16)
                theta_grid, theta_dot_grid = np.meshgrid(theta, theta_dot)
                state_grid = np.stack([np.cos(theta_grid), np.sin(theta_grid), theta_dot_grid], axis=-1)

                value_grid = np.array([self.V(state) for state in state_grid.reshape((16**2, 3))])
                value_grid = value_grid.reshape(theta_grid.shape)
                print(value_grid.shape)

                plt.imshow(value_grid, extent=[-np.pi, np.pi, -8, 8], origin='lower', aspect='auto')
                plt.title('Value Function')
                plt.xlabel('Theta')
                plt.ylabel('Angular Velocity')
                plt.show()

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
        action = mu + std * tf.random.normal(mean=0, stddev=1, shape=(self.output_size,))
        return action

    def _collect_trajectories(self, env):
        """Collects trajectories from the environment using the policy to select an action.
        It returns a TF tensor with the shape (num_trajectories, trajectory_length, state_size + action_size + reward_size)"""
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
                # trajectory.append(np.concatenate([np.array(state).flatten(), action[0], reward]))
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