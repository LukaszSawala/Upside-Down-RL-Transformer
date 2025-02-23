"""import gym
import d4rl # Import required to register environments, you may need to also import the submodule

# Create the environment
env = gym.make('maze2d-open-v0')

# d4rl abides by the OpenAI gym interface
env.reset()
env.step(env.action_space.sample())

# Each task is associated with a dataset
# dataset contains observations, actions, rewards, terminals, and infos
dataset = env.get_dataset()
print(dataset['observations']) # An N x dim_observation Numpy array of observations

# Alternatively, use d4rl.qlearning_dataset which
# also adds next_observations.
dataset = d4rl.qlearning_dataset(env)"""

""" TRIAL 2 import minari

dataset = minari.load_dataset("D4RL/antmaze/large-diverse-v1", download=True)"""

# TRIAL 3

import d4rl
import gymnasium as gym

# Make sure to use the CartPole environment
env = gym.make('CartPole-v1')

# Load the dataset (e.g., cartpole data from D4RL)
dataset = d4rl.qlearning.load('cartpole-medium-v0')