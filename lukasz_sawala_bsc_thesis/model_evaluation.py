import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models import NeuralNet
from scipy.stats import sem
from utils import parse_arguments

INPUT_SIZE = 105 + 2  # s_t + d_r and d_t
OUTPUT_SIZE = 8
NN_MODEL_PATH = "../models/best_nn_grid.pth"


def load_nn_model_for_eval(input_size: int, hidden_size: int,
                           output_size: int, checkpoint_path: str) -> NeuralNet:
    """
    Loads a Neural Network model from a given checkpoint path for evaluation.
    """
    model = NeuralNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model


def evaluate_get_rewards(env: gym.Env, model, d_h: float, d_r: float,
                         num_episodes: int = 1, max_episode_length: int = 1000) -> tuple:
    """
    Evaluate the performance of the model on the given environment.

    Args:
        env: The environment to evaluate on.
        model (NeuralNet): The model to evaluate.
        d_h: Desired horizon.
        d_r: Desired reward.
        num_episodes: The number of episodes to evaluate. Defaults to 1.
        max_episode_length: The maximum length of each episode. Defaults to 1000.
    Returns:
        float: The average reward over the specified number of episodes.
        list: A list of rewards for each episode.
    """
    episodic_reward_list = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        d_r_copy = d_r
        d_h_copy = d_h
        for _ in range(max_episode_length):
            obs_extended = np.concatenate((obs, [d_r_copy, d_h_copy]))
            obs_tensor = torch.tensor(obs_extended, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                actions = model(obs_tensor).squeeze(0).numpy()
            obs, reward, terminated, truncated, _ = env.step(actions)
            d_r_copy -= reward
            d_h_copy -= 1
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
    args = parse_arguments(training=False)

    if args["model_type"] == "NeuralNet":
        hidden_size = 256
        model = load_nn_model_for_eval(INPUT_SIZE, hidden_size, OUTPUT_SIZE, NN_MODEL_PATH)

    d_h = 1000.0
    d_r_options = [3000 + i * 200 for i in range(args["d_r_array_length"])]
    num_episodes = args["episodes"]

    env = gym.make("Ant-v5")  # render mode 'human' for visualization
    average_rewards = []
    sem_values = []

    for d_r in d_r_options:
        print("Trying with d_r:", d_r)
        _, episodic_rewards = evaluate_get_rewards(env, model, d_h, d_r, num_episodes=num_episodes)
        average_rewards.append(np.mean(episodic_rewards))
        sem_values.append(sem(episodic_rewards))

    plot_average_rewards(average_rewards, sem_values, d_r_options)
    env.close()
