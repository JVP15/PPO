import gym
import numpy as np
from a3_gym_env.envs.pendulum import CustomPendulumEnv

from PPO import PPOAgent

# Set USING_CUSTOM_ENVIRONMENT to True if you want to use the environment provided by the professor, and False
#   if you want to use Pendulum-v1 from OpenAI Gym. I think the only difference is that Pendulum-v1 stops after 200 timesteps
#   while the environment provided by the professor does not have a limit.

USING_CUSTOM_ENVIRONMENT = False

def state_helper(state):
    # state is [cos(theta), sin(theta), theta_dot], we want [theta, theta_dot]
    return [np.arctan2(state[1], state[0]), state[2]]

if USING_CUSTOM_ENVIRONMENT:
    env = CustomPendulumEnv()
else:
    env = gym.make('Pendulum-v1')

# sample hyperparameters
train_iterations = 10000
batch_size = 10000
epochs = 30
learning_rate = 1e-2
hidden_size = 8
n_layers = 2

agent = PPOAgent(input_size=2, output_size=1, log_std=-.5)

# get first observation of the environment and make sure the state is in the correct format
state = state_helper(env.reset())

for _ in range(train_iterations):
    # select action
    action = agent.action(state)

    # take action and get next observation, reward, done
    state, reward, done, _ = env.step(action)

    # if we're using a custom environment, we need to convert the state to [theta, theta_dot]
    state = state_helper(state)

    # render environment
    env.render()

    # check if episode is finished
    if done:
        # reset environment
        obs = env.reset()