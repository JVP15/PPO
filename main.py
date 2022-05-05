import gym
import numpy as np
import tensorflow as tf
from PPO import PPOAgent
import time

# Set USING_CUSTOM_ENVIRONMENT to True if you want to use the environment provided by the professor, and False
#   if you want to use Pendulum-v1 from OpenAI Gym. I think the only difference is that Pendulum-v1 stops after 200 timesteps
#   while the environment provided by the professor does not have a limit.

USING_CUSTOM_ENVIRONMENT = False

def state_helper(state):
    # state is [cos(theta), sin(theta), theta_dot], we want [theta, theta_dot]
    return [np.arctan2(state[1], state[0]), state[2]]

if USING_CUSTOM_ENVIRONMENT:
    from a3_gym_env.envs.pendulum import CustomPendulumEnv
    env = CustomPendulumEnv()
else:
    env = gym.make('Pendulum-v1')

# sample hyperparameters
total_timesteps = 20000
batch_size = 5
trajectory_length = 100
train_iterations = total_timesteps // (batch_size * trajectory_length)

# for visualization purposes, we'll use [theta, theta_dot],
#   but the agent will learn using [cos(theta), sin(theta), theta_dot]
agent = PPOAgent(input_size=3, output_size=1, log_std=-.01, num_trajectories=batch_size, trajectory_length=trajectory_length, learning_rate=1e-3)
agent.train(env=env, max_iterations=train_iterations, log_interval=1, eval_interval=5)

# check how the agent is doing after it's done training
test_episodes = 5
state = env.reset()

for _ in range(test_episodes):
    done = False
    print(f'Test Episode {_}')

    while not done:
        # select action
        action = agent.action(state)

        # take action and get next observation, reward, done
        state, reward, done, _ = env.step(action)

        # use the state helper to covert [cos(theta), sin(theta), theta_dot] to [theta, theta_dot]
        #state = state_helper(state)

        # render environment
        env.render()

        # check if episode is finished
        if done:
            # reset environment
            obs = env.reset()



