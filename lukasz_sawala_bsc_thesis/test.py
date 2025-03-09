# Dataset testing
import h5py
import os
import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)
sys.path.append(parent_dir)


file_path = "data/external/main_data.hdf5"

with h5py.File(file_path, "r") as f:
    episode_87 = f["episode_87"]

    # Print all the keys inside episode_87 (actions, infos, etc.)
    print("Keys in episode_87:", list(episode_87.keys()))

    # Inspect data in each attribute
    actions = episode_87["actions"][:]
    infos = episode_87["infos"]
    observations = episode_87["observations"][:]
    rewards = episode_87["rewards"][:]
    terminations = episode_87["terminations"][:]
    truncations = episode_87["truncations"][:]

    # Print some sample data
    print("Actions sample:", actions[:5])
    print("Observations sample:", observations[:5])
    print("Rewards sample:", rewards[:5])

    # Dimensions:
    print(f"Action dimensions: {actions.shape}")
    print(f"Observation dimensions: {observations.shape}")
    print(f"Reward dimensions: {rewards.shape}")
    print(f"Termination dimensions: {terminations.shape}")
    print(f"Truncation dimensions: {truncations.shape}")
