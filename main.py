import gym
import numpy as np
import tensorflow as tf
from PPO import PPOAgent
import time
import matplotlib.pyplot as plt

# Set USING_CUSTOM_ENVIRONMENT to True if you want to use the environment provided by the professor, and False
#   if you want to use Pendulum-v1 from OpenAI Gym. I think the only difference is that Pendulum-v1 stops after 200 timesteps
#   while the environment provided by the professor does not have a limit.

USING_CUSTOM_ENVIRONMENT = False

if USING_CUSTOM_ENVIRONMENT:
    from a3_gym_env.envs.pendulum import CustomPendulumEnv
    env = CustomPendulumEnv()
else:
    env = gym.make('Pendulum-v1')

# sample hyperparameters
total_timesteps = 1000
batch_size = 5
trajectory_length = 100
train_iterations = total_timesteps // (batch_size * trajectory_length)

agent = PPOAgent(input_size=3, output_size=1, log_std=-.01, num_trajectories=batch_size, trajectory_length=trajectory_length, learning_rate=1e-3)
agent.train(env=env, max_iterations=train_iterations, log_interval=1)

# once we are done training the agent, display the reward to go plot as a function of iteration
iterations = np.arange(train_iterations)
plt.plot(iterations, agent.rewards)
plt.xlabel('Iteration')
plt.ylabel('Reward to Go')
plt.title('Reward to Go vs. Iteration')
plt.show()

# then, plot the loss as a function of iteration
iterations = np.arange(0, len(agent.losses), 1)
plt.plot(iterations, agent.losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss vs. Iteration')
plt.show()

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



