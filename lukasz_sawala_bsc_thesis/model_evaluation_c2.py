import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models import NeuralNet
from scipy.stats import sem
from utils import parse_arguments
from transformers import DecisionTransformerModel, DecisionTransformerConfig
from collections import deque
from zeus.monitor import ZeusMonitor


INPUT_SIZE = 105 + 2  # s_t + d_r and d_t
OUTPUT_SIZE = 8
NN_MODEL_PATH = "../models/best_nn_grid.pth"
DT_MODEL_PATH = "../models/best_DT_grid.pth"
MAX_LENGTH = 60
STATE_DIM = INPUT_SIZE - 2  # used for the DT


def load_nn_model_for_eval(input_size: int, hidden_size: int,
                           output_size: int, checkpoint_path: str,
                           device: str) -> NeuralNet:
    """
    Loads a Neural Network model from a given checkpoint path for evaluation.
    """
    model = NeuralNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    model.eval()
    return model


def load_dt_model_for_eval(state_dim: int, act_dim: int, max_length: int,
                           checkpoint_path: str,
                           device: str) -> DecisionTransformerModel:
    """
    Loads a Decision Transformer model from a given checkpoint path for evaluation.
    """
    config = DecisionTransformerConfig(state_dim=state_dim, act_dim=act_dim, max_length=max_length)
    model = DecisionTransformerModel(config)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    model.eval()
    return model


def evaluate_get_rewards(env: gym.Env, model, d_h: float, d_r: float,
                         num_episodes: int = 1, max_episode_length: int = 1000,
                         model_type: str = "NeuralNet",
                         device: str = "cpu") -> tuple:
    """
    Evaluate the performance of the model on the given environment.

    Args:
        env: The environment to evaluate on.
        model (NeuralNet): The model to evaluate.
        d_h: Desired horizon.
        d_r: Desired reward.
        num_episodes: The number of episodes to evaluate. Defaults to 1.
        max_episode_length: The maximum length of each episode. Defaults to 1000.
        model_type: The type of model to evaluate. Can be "NeuralNet" or "DecisionTransformer".
    Returns:
        float: The average reward over the specified number of episodes.
        list: A list of rewards for each episode.
    """
    if model_type == "NeuralNet":
        return _evaluate_neural_net(env, model, d_h, d_r, num_episodes, max_episode_length)
    elif model_type == "DecisionTransformer":
        return _evaluate_decision_transformer(env, model, d_r, num_episodes, max_episode_length, device)


def _evaluate_neural_net(env: gym.Env, model, d_h: float, d_r: float,
                         num_episodes: int, max_episode_length: int) -> tuple:
    """
    Evaluate the performance of the Neural Network model on the given environment.
    """
    episodic_rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        d_r_copy, d_h_copy = d_r, d_h
        total_reward = 0
        for _ in range(max_episode_length):
            obs_input = torch.tensor(np.concatenate((obs, [d_r_copy, d_h_copy])), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = model(obs_input).squeeze(0).numpy()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            d_r_copy -= reward
            d_h_copy -= 1
            if terminated or truncated:
                break
        episodic_rewards.append(total_reward)
    print("max-min reward for this dr:", max(episodic_rewards), "-", min(episodic_rewards))
    return np.mean(episodic_rewards), episodic_rewards


def _evaluate_decision_transformer(env: gym.Env, model, d_r: float,
                                   num_episodes: int, max_episode_length: int, device) -> tuple:
    """
    Evaluate the performance of the Decision Transformer model on the given environment.
    """
    episodic_rewards = []
    act_dim = model.config.act_dim  # Get action dimension from model config
    state_dim = model.config.state_dim  # Get state dimension from model config

    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        target_return = float(d_r)  # Initial target return

        # Use deques to efficiently manage history
        state_history = deque([np.zeros(state_dim, dtype=np.float32)] * MAX_LENGTH, maxlen=MAX_LENGTH)
        action_history = deque([np.ones(act_dim, dtype=np.float32) * -10.0] * MAX_LENGTH, maxlen=MAX_LENGTH)
        rtg_history = deque([0.0] * MAX_LENGTH, maxlen=MAX_LENGTH)  # Store scalar RTGs
        timestep_history = deque([0] * MAX_LENGTH, maxlen=MAX_LENGTH)

        # Add initial state
        state_history.append(obs.astype(np.float32))
        rtg_history.append(target_return)
        # timestep_history correctly starts with 0 from initialization

        for t in range(max_episode_length):
            # --- Prepare model inputs ---
            states = np.array(state_history, dtype=np.float32)
            actions = np.array(action_history, dtype=np.float32)
            rtgs = np.array(rtg_history, dtype=np.float32).reshape(-1, 1)
            timesteps = np.array(timestep_history, dtype=np.int64)
            # --------------------------------

            current_len = min(t + 1, MAX_LENGTH)
            mask = np.concatenate([np.zeros(MAX_LENGTH - current_len), np.ones(current_len)], dtype=np.float32)

            # Convert to tensors and add batch dimension
            states_tensor = torch.tensor(states, dtype=torch.float32).unsqueeze(0).to(device)
            actions_tensor = torch.tensor(actions, dtype=torch.float32).unsqueeze(0).to(device)
            rtgs_tensor = torch.tensor(rtgs, dtype=torch.float32).unsqueeze(0).to(device)
            timesteps_tensor = torch.tensor(timesteps, dtype=torch.long).unsqueeze(0).to(device)
            mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                model_outputs = model(
                    states=states_tensor,
                    actions=actions_tensor,
                    returns_to_go=rtgs_tensor,
                    timesteps=timesteps_tensor,
                    attention_mask=mask_tensor,
                    return_dict=True  # Use dictionary output for clarity
                )
                action_preds = model_outputs['action_preds']  # model_outputs[0] if return_dict=False

                # Extract the action prediction for the last timestep in the input sequence
                action = action_preds[0, -1].cpu().numpy()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            # Update history for the next iteration
            action_history.append(action.astype(np.float32))
            state_history.append(obs.astype(np.float32))
            current_rtg = rtg_history[-1]
            rtg_history.append(current_rtg - reward)
            timestep_history.append(t + 1)

            if terminated or truncated:
                break

        episodic_rewards.append(total_reward)

    return np.mean(episodic_rewards), episodic_rewards


def plot_average_rewards(average_rewards: list, sem_values: list,
                         d_r_values: list, title="Average Reward vs. d_r",
                         save_path: str = "average_rewards_plot.png"):
    """
    Plots the average rewards for different values of d_r with standard error bars.
    """
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.figure(figsize=(12, 7))
    sns.lineplot(x=d_r_values, y=average_rewards, linewidth=2.5, color="royalblue", marker="o", label="Average Reward")
    plt.fill_between(d_r_values, np.array(average_rewards) - np.array(sem_values), np.array(average_rewards) + np.array(sem_values), color='royalblue', alpha=0.2)
    plt.plot(d_r_values, d_r_values, linestyle="dotted", color="gray")
    plt.xlabel("d_r", fontsize=12)
    plt.ylabel("Average Reward", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    sns.despine()
    plt.savefig(save_path)
    print(f"Average rewards plot saved in {save_path}")


if __name__ == "__main__":
    args = parse_arguments(training=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("starting evaluation for args:", args, "device:", device)
    if args["model_type"] == "NeuralNet":
        hidden_size = 256
        model = load_nn_model_for_eval(INPUT_SIZE, hidden_size, OUTPUT_SIZE, NN_MODEL_PATH, device)
    elif args["model_type"] == "DecisionTransformer":
        model = load_dt_model_for_eval(STATE_DIM, OUTPUT_SIZE, MAX_LENGTH, DT_MODEL_PATH, device)
    else:
        raise ValueError(f"Unsupported model_type: {args['model_type']}")

    d_h = 1000.0
    d_r_options = [2000 + i * 100 for i in range(args["d_r_array_length"])]
    num_episodes = args["episodes"]

    env = gym.make("Ant-v5")  # render mode 'human' for visualization
    average_rewards = []
    sem_values = []

    #monitor = ZeusMonitor(gpu_indices=[0] if device.type == 'cuda' else [], cpu_indices=[0, 1])
    #monitor.begin_window(f"evaluation_{args['model_type']}")

    for d_r in d_r_options:
        print('=' * 50)
        print("Trying with d_r:", d_r)
        _, episodic_rewards = evaluate_get_rewards(env, model, d_h, d_r,
                                                   num_episodes=num_episodes,
                                                   model_type=args["model_type"],
                                                   device=device)
        average_rewards.append(np.mean(episodic_rewards))
        sem_values.append(sem(episodic_rewards))

    #mes = monitor.end_window(f"evaluation_{args['model_type']}")
    #print(f"Training grid search took {mes.time} s and consumed {mes.total_energy} J.")

    save_path = f"average_rewards_plot_{args['model_type']}.png"
    plot_average_rewards(average_rewards, sem_values, d_r_options, save_path=save_path)
    env.close()
