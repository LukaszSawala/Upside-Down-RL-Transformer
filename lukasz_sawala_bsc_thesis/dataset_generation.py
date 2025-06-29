import gymnasium as gym
import torch
import numpy as np
import gymnasium_robotics
import h5py
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from models import AntMazeBERTPretrainedMazeWrapper, AntMazeNNPretrainedMazeWrapper
from transfer_eval_main import extract_goal_direction, load_antmaze_bertmlp_model_for_eval, load_antmaze_nn_model_for_eval

# --- Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INITIAL_ANTMAZE_BERT_PATH = "../models/cond4_berttiny-18_512.pth"
NEW_BERT_MODEL_PATH = "cond5-berttiny-18-512.pth"
INITIAL_ANTMAZE_NN_PATH = "../models/cond4_NN-18_512.pth"
NEW_NN_MODEL_PATH = "cond5-NN-18-512.pth"

OUTPUT_HDF5_PATH = "antmaze_rollout_current_dataset.hdf5"


def generate_dataset(d_h: float, d_r_options: list, num_episodes_per_dr: int, start_from_condition4: bool,
                     retain_best_previous_data: bool = False, model_name="ANTMAZE_BERT_MLP"):
    """
    Generates a dataset of trajectories for the AntMaze environment using a pretrained model.
    Args:
        d_h (float): Initial horizon value used for episode termination.
        d_r_options (list): List of reward threshold values to test during data collection.
        num_episodes_per_dr (int): Number of episodes to collect for each reward threshold.
        start_from_condition4 (bool): Flag to determine if data collection starts from condition 4.
        retain_best_previous_data (bool, optional): If True, retains the best data from previous collections.
        model_name (str, optional): Name of the model to use for data generation. Defaults to "ANTMAZE_BERT_MLP".

    This function collects data by running episodes in the AntMaze environment with a specified model.
    It collects observations, actions, rewards, and goal vectors for each episode, and processes them
    into a dataset. The resulting dataset is saved to an HDF5 file, and a 2D histogram of reward-to-go
    versus horizon is generated and saved as a PNG image.
    """
    # --- Load environment and model ---
    gym.register_envs(gymnasium_robotics)
    env = gym.make("AntMaze_MediumDense-v5")
    if model_name == "ANTMAZE_BERT_MLP":
        if start_from_condition4:
            model_components = load_antmaze_bertmlp_model_for_eval(INITIAL_ANTMAZE_BERT_PATH, DEVICE)
            model = AntMazeBERTPretrainedMazeWrapper(*model_components).to(DEVICE)
        else:
            model_components = load_antmaze_bertmlp_model_for_eval("", DEVICE, initialize_from_scratch=True)
            model = AntMazeBERTPretrainedMazeWrapper(*model_components).to(DEVICE)
            checkpoint = torch.load(NEW_BERT_MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(checkpoint["model"])   # initialize from scratch to be able to load the model properly
    else:  # NN model
        if start_from_condition4:
            model = load_antmaze_nn_model_for_eval(INITIAL_ANTMAZE_NN_PATH, DEVICE)
            model = AntMazeNNPretrainedMazeWrapper(model).to(DEVICE)
        else:
            model = load_antmaze_nn_model_for_eval("", DEVICE,
                                                   initialize_from_scratch=True)
            model = AntMazeNNPretrainedMazeWrapper(model).to(DEVICE)
            checkpoint = torch.load(NEW_NN_MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(checkpoint["model"])

    model.eval()

    # --- Parameters ---
    state_dim = 27  # AntMaze uses reduced state space

    # --- Data storage for all episodes ---
    all_actions = []
    all_observations = []
    all_rewards_to_go = []
    all_time_to_go = []
    all_goal_vectors = []

    # --- Main Data Collection Loop ---
    episode_count = 0
    low_reward_episodes = 0
    for d_r in d_r_options:
        print(f"\nCollecting data for d_r = {d_r} with d_h = {d_h}...")
        for ep in range(num_episodes_per_dr):
            # print(f"Collecting: [d_r={d_r}] Episode {ep + 1}/{num_episodes_per_dr}")
            obs, _ = env.reset()
            d_h_copy = d_h
            d_r_copy = d_r
            done = False

            # Store data for the current episode temporarily
            episode_observations = []
            episode_actions = []
            episode_rewards = []
            episode_goal_vectors = []
            obtained_return = 0.0
            while not done and d_h_copy > 0:
                goal_vec = extract_goal_direction(obs)
                state = obs["observation"][:state_dim]

                with torch.no_grad():
                    action_tensor = model(state, d_r_copy, d_h_copy, goal_vec, DEVICE, use_goal=True)
                action = action_tensor.squeeze(0).cpu().numpy()

                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                obtained_return += reward

                # Append step data to temporary lists
                episode_observations.append(state)
                episode_actions.append(action)
                episode_rewards.append(float(reward))
                episode_goal_vectors.append(goal_vec)

                d_r_copy -= reward
                d_h_copy -= 1

            # --- On-the-fly Processing (after each episode) ---
            if obtained_return < 2.0 and d_r > 0:
                low_reward_episodes += 1
                if low_reward_episodes % 7 != 0:
                    continue  # keep ~14% of low reward episodes

            # Convert episode lists to NumPy arrays
            rewards_np = np.array(episode_rewards)

            # 1. Calculate rewards-to-go
            # This is a vectorized and efficient way to compute the cumulative sum of future rewards
            rewards_to_go = np.cumsum(rewards_np[::-1])[::-1]

            # 2. Calculate time-to-go
            # This creates an array like [T, T-1, ..., 1] where T is the episode length
            time_to_go = np.arange(len(rewards_np), 0, -1)

            # Append processed data to the master lists
            all_observations.append(np.array(episode_observations))
            all_actions.append(np.array(episode_actions))
            all_goal_vectors.append(np.array(episode_goal_vectors))
            all_rewards_to_go.append(rewards_to_go)
            all_time_to_go.append(time_to_go)

            episode_count += 1

    print(f"\nCollected {episode_count} episodes. Now concatenating and saving...")

    # --- Final Concatenation and Saving ---

    # Concatenate all episodes into single large NumPy arrays
    final_observations = np.concatenate(all_observations, axis=0).astype(np.float32)
    final_actions = np.concatenate(all_actions, axis=0).astype(np.float32)
    final_goal_vectors = np.concatenate(all_goal_vectors, axis=0).astype(np.float32)
    final_rewards_to_go = np.concatenate(all_rewards_to_go, axis=0).astype(np.float32)
    final_time_to_go = np.concatenate(all_time_to_go, axis=0).astype(np.int32)

    print("stats before concatenation:", 48 * "=")
    print("AVERAGE REWARD TO GO:", np.mean(final_rewards_to_go))  # the higher this is, the more high reward trajectories we obtained
    print("AVERAGE OBTAINED REWARD PER EPISODE:", np.mean([rewards[0] for rewards in all_rewards_to_go]))
    print(50 * "=")

    if retain_best_previous_data and not start_from_condition4:
        # load previous data if the previous model was already finetuned before (start_from_condition4=False)
        with h5py.File(OUTPUT_HDF5_PATH, "r") as f:
            observations = f["concatenated_data"]["observations"][:]
            actions = f["concatenated_data"]["actions"][:]
            goal_vectors = f["concatenated_data"]["goal_vector"][:]
            rewards_to_go = f["concatenated_data"]["rewards_to_go"][:]
            time_to_go = f["concatenated_data"]["time_to_go"][:]

            mask = rewards_to_go.squeeze() > 500
            # Stack the filtered data
            filtered_observations = observations[mask]
            filtered_actions = actions[mask]
            filtered_goal_vectors = goal_vectors[mask]
            filtered_rewards_to_go = rewards_to_go[mask]
            filtered_time_to_go = time_to_go[mask]

            # Concatenate them to the newly collected data
            final_observations = np.concatenate([final_observations, filtered_observations], axis=0)
            final_actions = np.concatenate([final_actions, filtered_actions], axis=0)
            final_goal_vectors = np.concatenate([final_goal_vectors, filtered_goal_vectors], axis=0)
            final_rewards_to_go = np.concatenate([final_rewards_to_go, filtered_rewards_to_go], axis=0)
            final_time_to_go = np.concatenate([final_time_to_go, filtered_time_to_go], axis=0)

    # make a 2d histogram of dr and dt to go to see how they are related
    df = pd.DataFrame({"Reward-to-Go": final_rewards_to_go, "Horizon": final_time_to_go})
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="Reward-to-Go", y="Horizon", bins=50, cmap="viridis", cbar=True)
    plt.title("2D Histogram: Reward-to-Go vs Horizon", fontsize=18, weight="bold")
    plt.xlabel("Reward-to-Go", fontsize=14, weight="bold")
    plt.ylabel("Horizon", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig("reward_to_go_vs_horizon_newdata.png")

    # Save the concatenated data to a single HDF5 file
    with h5py.File(OUTPUT_HDF5_PATH, "w") as f:
        # Using a group is good practice for organizing data within the HDF5 file
        data_group = f.create_group("concatenated_data")
        data_group.create_dataset("observations", data=final_observations)
        data_group.create_dataset("actions", data=final_actions)
        data_group.create_dataset("rewards_to_go", data=final_rewards_to_go)
        data_group.create_dataset("time_to_go", data=final_time_to_go)
        data_group.create_dataset("goal_vector", data=final_goal_vectors)
        # check shapes
        print("Shapes of saved datasets:")
        print(f"Observations: {final_observations.shape}")
        print(f"Actions: {final_actions.shape}")
        print(f"Rewards-to-Go: {final_rewards_to_go.shape}")
        print(f"Time-to-Go: {final_time_to_go.shape}")
        print(f"Goal Vectors: {final_goal_vectors.shape}")

    print(f"Data processing complete. Final dataset saved to {OUTPUT_HDF5_PATH}")


if __name__ == "__main__":
    d_h = 1000.0
    d_r_options = [i * 50 for i in range(21)]
    num_episodes_per_dr = 20
    start_from_condition4 = True  # Set to True at the beggining of the loop
    generate_dataset(d_h=d_h, d_r_options=d_r_options,
                     num_episodes_per_dr=num_episodes_per_dr, start_from_condition4=start_from_condition4)
