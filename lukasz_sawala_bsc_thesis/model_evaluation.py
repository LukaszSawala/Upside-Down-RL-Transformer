import torch.nn as nn
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from models import NeuralNet
from scipy.stats import sem

def load_model_for_eval(input_size: int, hidden_size: int, 
                        output_size: int, checkpoint_path: str) -> NeuralNet:
    """
    Loads a Neural Network model from a given checkpoint path for evaluation.
    """
    model = NeuralNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model


def evaluate_get_rewards(env: gym.Env, model, d_t: float, d_r: float, 
                         num_episodes=1, max_episode_length=1000) -> tuple:
    """
    Evaluate the performance of the model on the given environment.

    Args:
        env (gym.Env): The environment to evaluate on.
        model (NeuralNet): The model to evaluate.
        d_t (float): The initial distance to the target.
        d_r (float): The initial distance to the reward.
        num_episodes (int, optional): The number of episodes to evaluate. Defaults to 1.
        max_episode_length (int, optional): The maximum length of each episode. Defaults to 1000.

    Returns:
        float: The average reward over the specified number of episodes.
        list: A list of rewards for each episode.
    """
    episodic_reward_list = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        d_r_copy = d_r
        d_t_copy = d_t
        for _ in range(max_episode_length):
            obs_extended = np.concatenate((obs, [d_r_copy, d_t_copy]))
            obs_tensor = torch.tensor(obs_extended, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                actions = model(obs_tensor).squeeze(0).numpy()
            obs, reward, terminated, truncated, _ = env.step(actions)
            d_r_copy -= reward
            d_t_copy -= 1
            episode_reward += reward

            # Render environment
            # env.render()
            # time.sleep(0.05)

            if terminated or truncated:
                break
        episodic_reward_list.append(episode_reward)
    env.close()
    return np.mean(episodic_reward_list), episodic_reward_list

def plot_average_rewards(average_rewards: list, sem_values: list, 
                         d_r_values: list, title="Average Reward vs. d_r",
                         save_path="average_rewards_plot.png"):
    """
    Plots the average rewards for different values of d_r with standard error bars.
    """
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=d_r_values, y=average_rewards, linewidth=2.5, color="royalblue", marker="o", label="Average Reward")
    plt.fill_between(d_r_values, np.array(average_rewards) - np.array(sem_values), np.array(average_rewards) + np.array(sem_values), color='royalblue', alpha=0.2)
    plt.xlabel("d_r", fontsize=12)
    plt.ylabel("Average Reward", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    sns.despine()
    plt.savefig(save_path)
    print(f"Average rewards plot saved in {save_path}")


if __name__ == "__main__":
    input_size = 105 + 2  # s_t + d_r and d_t
    hidden_size = 256
    output_size = 8
    checkpoint_path = "../models/best_nn_grid.pth"
    model = load_model_for_eval(input_size, hidden_size, output_size, checkpoint_path)

    d_t = 1000.0
    d_r_options = [3000 + i * 200 for i in range(11)]
    num_episodes = 10
    num_trials = 5

    env = gym.make("Ant-v5") # render mode 'human' for visualization
    average_rewards = []
    sem_values = []
    for d_r in d_r_options:
        print("Trying with d_r:", d_r)
        trial_rewards = []
        for _ in range(num_trials):
            avg_reward, _ = evaluate_get_rewards(env, model, d_t, d_r, num_episodes=num_episodes)
            trial_rewards.append(avg_reward)
        average_rewards.append(np.mean(trial_rewards))
        sem_values.append(sem(trial_rewards))
    plot_average_rewards(average_rewards, sem_values, d_r_options)
    env.close()
