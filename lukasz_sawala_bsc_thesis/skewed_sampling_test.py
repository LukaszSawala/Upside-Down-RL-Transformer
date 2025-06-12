# Dataset testing
import h5py
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta
from torch.utils.data import Dataset


INPUT_SIZE = 105 + 2  # s_t + d_r and d_t
OUTPUT_SIZE = 8
CONCATENATED_DATA_PATH = "../data/processed/concatenated_data.hdf5"


def load_data(data_path: str) -> list:
    """
    Load episodic data from an HDF5 file.
    Parameters:
        data_path (str): The path to the HDF5 file.
    Returns:
        list: A list where each element is a tuple of:
            - observations (np.ndarray of shape (T, obs_dim))
            - actions (np.ndarray of shape (T, action_dim))
            - rewards_to_go (np.ndarray of shape (T, 1))
            - time_to_go (np.ndarray of shape (T, 1))
    """
    episodic_data = []

    with h5py.File(data_path, "r") as f:
        data_group = f["episodic_data"]  # Root group containing episodes

        for episode_key in data_group.keys():
            episode = data_group[episode_key]

            observations = episode["observations"][:]
            actions = episode["actions"][:]
            rewards_to_go = episode["rewards_to_go"][:].reshape(-1, 1)
            time_to_go = episode["time_to_go"][:].reshape(-1, 1)

            episodic_data.append((observations, actions, rewards_to_go, time_to_go))

    return episodic_data


class TrajectoryDataset(Dataset):
    def __init__(self, episodic_data, context_window_size, min_sample_size=None, skew_factor=1):
        """
        Args:
            episodic_data (list): A list of episodes, where each episode is a tuple of (obs, actions, rewards_to_go, time_to_go).
            context_window_size (int): The max length of sampled sequences.
            min_sample_size (int, optional): Minimum sequence length (default: half of context_window_size).
        """
        self.episodic_data = episodic_data
        self.context_window_size = context_window_size
        self.min_sample_size = min_sample_size or (context_window_size // 2)
        self.skew_factor = skew_factor

    def __len__(self):
        return len(self.episodic_data)  # Number of episodes

    def sample_window_size(self):
        """
        Sample a sequence length using a right-skewed Beta distribution.
        """
        beta_sample = np.random.beta(5, self.skew_factor)  # Beta(5, k=1) skews toward 1
        sampled_length = int(self.min_sample_size + beta_sample * (self.context_window_size - self.min_sample_size))
        return min(sampled_length, self.context_window_size)

    def __getitem__(self, index: int):
        """
        Samples a random variable-length trajectory from a randomly selected episode.
        """
        obs, actions, rewards_to_go, time_to_go = self.episodic_data[index]

        # Pick a biased random window size
        window_size = self.sample_window_size()

        # Ensure we have enough data to sample
        if len(obs) < window_size:
            raise ValueError(f"Episode {index} is too short for window size {window_size}")

        # Select a random starting point
        start_idx = random.randint(0, len(obs) - window_size)

        # Extract the sequence
        obs_sample = obs[start_idx:start_idx + window_size]
        actions_sample = actions[start_idx:start_idx + window_size]
        rewards_sample = rewards_to_go[start_idx:start_idx + window_size]
        time_sample = time_to_go[start_idx:start_idx + window_size]

        # Convert to tensors
        obs_tensor = torch.tensor(obs_sample, dtype=torch.float32)
        actions_tensor = torch.tensor(actions_sample, dtype=torch.float32)
        rewards_tensor = torch.tensor(rewards_sample, dtype=torch.float32)
        time_tensor = torch.tensor(time_sample, dtype=torch.float32)

        # Return as dictionary (easier handling later)
        return {
            "observations": obs_tensor,
            "actions": actions_tensor,
            "rewards_to_go": rewards_tensor,
            "time_to_go": time_tensor
        }


def plot_beta_distribution(a=5, b=1, num_samples=10000, title="Beta Distribution", save_path="beta_distribution_plot.png"):
    """
    Plots the Beta distribution for given parameters a and b with a style similar to the `plot_average_rewards`.

    Args:
        a (float): Alpha parameter of the Beta distribution.
        b (float): Beta parameter of the Beta distribution.
        num_samples (int): Number of samples to plot.
        title (str): Title of the plot.
        save_path (str): Path to save the plot.
    """
    x = np.linspace(0, 1, num_samples)
    y = beta.pdf(x, a, b)
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.figure(figsize=(12, 7))
    sns.lineplot(x=x, y=y, linewidth=2.5, color="royalblue", marker="o", label=f"Beta({a}, {b})")
    plt.fill_between(x, y, color='royalblue', alpha=0.2)
    plt.plot(x, x, linestyle="dotted", color="gray", label="y = x")
    plt.xlabel("Sampled Value (Normalized)", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    sns.despine()
    plt.legend()
    plt.savefig(save_path)
    print(f"Beta distribution plot saved in {save_path}")


if __name__ == "__main__":
    plot_beta_distribution(5, 1,
                           title="Right-Skewed Beta Distribution",
                           save_path="beta_right_skewed.png")
