import gym
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# env = gym.make('Pendulum-v1-custom')
env = gym.make('Pendulum-v1')

# sample hyperparameters
train_iterations = 10000
batch_size = 10000
epochs = 30
learning_rate = 1e-2
hidden_size = 8
n_layers = 2

print(env.action_space)

# get first observation of the environment
obs = env.reset()

for _ in range(train_iterations):
    # select action
    action = env.action_space.sample()

    # take action and get next observation, reward, done
    obs, reward, done, _ = env.step(action)

    # render environment
    env.render()

    # check if episode is finished
    if done:
        # reset environment
        obs = env.reset()