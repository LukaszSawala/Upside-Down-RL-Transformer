import torch.nn as nn
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc5 = nn.Linear(hidden_size // 4, output_size)
        self.relu = nn.ReLU()  # ReLU activation function

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = torch.tanh(x)  # Enforces the action range between -1 and 1
        return x


def load_model_for_eval(input_size: int, hidden_size: int, output_size: int, checkpoint_path: str) -> NeuralNet:
    """
    Loads a model from a checkpoint path and puts it into evaluation mode.

    Parameters:
    input_size (int): The size of the input layer.
    hidden_size (int): The size of the hidden layer.
    output_size (int): The size of the output layer.
    checkpoint_path (str): The path to the model checkpoint.

    Returns:
    NeuralNet: The loaded model in evaluation mode.
    """
    model = NeuralNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model


def evaluate_get_rewards(env, model, d_t, d_r, num_episodes=1, max_episode_length=1000):
    raw_reward_list = []
    episodic_reward_list = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        d_r_copy = d_r
        d_t_copy = d_t
        for _ in range(max_episode_length):
            # Convert observation to tensor while adding d_r and d_t
            obs_extended = np.concatenate((obs, [d_r_copy, d_t_copy]))
            obs_tensor = torch.tensor(obs_extended, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                actions = model(obs_tensor).squeeze(0).numpy()

            # print(obs)
            # print(actions)
            obs, reward, terminated, truncated, _ = env.step(actions)

            d_r_copy -= reward
            d_t_copy -= 1
            episode_reward += reward
            raw_reward_list.append(reward)

            # Render environment
            #env.render()
            # time.sleep(0.05) uncomment to see whats up
            if terminated or truncated:
                break
        episodic_reward_list.append(episode_reward)
        print(f"episode finished with a reward of {episode_reward:.2f}")

    env.close()
    total_reward = sum(episodic_reward_list)
    print(f"Total reward obtained: {total_reward:.2f}")
    return raw_reward_list, episodic_reward_list


def plot_rewards(reward_list, title="Rewards Over Time", save_path="rewards_plot.png"):
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.figure(figsize=(10, 5))
    x_values = np.arange(len(reward_list))

    # Plot the rewards
    sns.lineplot(x=x_values, y=reward_list, linewidth=2.5, color="royalblue", label="Reward")

    # Plot smoothed rewards
    window_size = 5
    smoothed_rewards = pd.Series(reward_list).rolling(window=window_size, min_periods=1, center=True).mean()
    sns.lineplot(x=x_values, y=smoothed_rewards, linestyle="dashed", color="crimson", label="Smoothed Reward (n=5)")

    plt.xlabel("Time Step / Episode", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")

    # Remove top & right borders for a cleaner look
    sns.despine()
    plt.savefig(save_path)
    print(f"Rewards plot saved in {save_path}")


if __name__ == "__main__":
    input_size = 105 + 2  # s_t + d_r and d_t
    hidden_size = 256
    output_size = 8
    checkpoint_path = "best_nn.pth"
    model = load_model_for_eval(input_size, hidden_size, output_size, checkpoint_path)

    d_t = 1000.0  # ?
    d_r = 4400.0  # avg from the dataset

    # create the env and run evaluation
    num_episodes = 5
    env = gym.make("Ant-v5")  # 'human' for visualization
    _, episode_rewards = evaluate_get_rewards(env, model, d_t, d_r, num_episodes=10)
    plot_rewards(episode_rewards, title="Episodic Rewards Over Time", save_path="rewards_plot.png")
