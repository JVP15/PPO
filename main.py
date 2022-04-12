import gym
import numpy as np

from PPO import PPOAgent

# If you haven't registered the custom environment (like me), set this to False. If it's false, the environment will be
#   set to Pendulum-v1, which I think is similar to the custom pendulum environment except its state space is
#   [cos(theta), sin(theta), theta_dot] instead of [theta, theta_dot].
# The state_helper functon will automatically take care of this though

USING_CUSTOM_ENVIRONMENT = False

def state_helper(state):
    if USING_CUSTOM_ENVIRONMENT:
        return state
    else:
        # state is [cos(theta), sin(theta), theta_dot], we want [theta, theta_dot]
        return [np.arccos(state[0]), state[2]]

env_name = 'Pendulum-v1-custom' if USING_CUSTOM_ENVIRONMENT else 'Pendulum-v1'

env = gym.make(env_name)

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