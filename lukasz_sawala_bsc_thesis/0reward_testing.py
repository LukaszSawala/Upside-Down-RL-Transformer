import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
from transformers import DecisionTransformerConfig, DecisionTransformerModel

from models import NeuralNet

INPUT_SIZE = 105 + 2  # s_t + d_r and d_t
OUTPUT_SIZE = 8
NN_MODEL_PATH = "../models/best_nn_grid.pth"
DT_MODEL_PATH = "../models/best_DT_grid.pth"
MAX_LENGTH = 60
STATE_DIM = INPUT_SIZE - 2  # used for the DT


def load_nn_model_for_eval(
    input_size: int,
    hidden_size: int,
    output_size: int,
    checkpoint_path: str,
    device: str = "cpu",
) -> NeuralNet:
    """Loads a Neural Network model for evaluation."""
    model = NeuralNet(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_dt_model_for_eval(
    state_dim: int,
    act_dim: int,
    max_length: int,
    checkpoint_path: str,
    device: str = "cpu",
) -> DecisionTransformerModel:
    """Loads a Decision Transformer model for evaluation."""
    config = DecisionTransformerConfig(
        state_dim=state_dim, act_dim=act_dim, max_length=max_length
    )
    model = DecisionTransformerModel(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def _run_nn_low_reward_test(
    env: gym.Env,
    model: NeuralNet,
    d_r: float,
    d_h: float,
    device: str,
    time_interval: float,
):
    """Runs the low reward test for a Neural Network model."""
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    while not done or d_h > 0:
        obs_input = (
            torch.tensor(np.concatenate((obs, [d_r, d_h])), dtype=torch.float32)
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():
            action = model(obs_input).squeeze(0).cpu().numpy()

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        d_r -= reward
        d_h -= 1

        print(
            f"Current Reward: {total_reward:.2f} | Desired Reward Left: {d_r:.2f} | Desired Horizon Left: {d_h:.2f}"
        )
        env.render()
        time.sleep(time_interval)

        if terminated or truncated:
            done = True


def _run_dt_low_reward_test(
    env: gym.Env,
    model: DecisionTransformerModel,
    d_r: float,
    d_h: float,
    device: str,
    time_interval: float,
):
    """Runs the low reward test for a Decision Transformer model."""
    act_dim = model.config.act_dim
    state_dim = model.config.state_dim
    state_history = deque(
        [np.zeros(state_dim, dtype=np.float32)] * MAX_LENGTH, maxlen=MAX_LENGTH
    )
    action_history = deque(
        [np.ones(act_dim, dtype=np.float32) * -10.0] * MAX_LENGTH, maxlen=MAX_LENGTH
    )
    rtg_history = deque([d_r] * MAX_LENGTH, maxlen=MAX_LENGTH)
    timestep_history = deque([0] * MAX_LENGTH, maxlen=MAX_LENGTH)

    obs, _ = env.reset()
    state_history.append(obs.astype(np.float32))
    total_reward = 0.0

    for t in range(int(d_h)):
        states = np.array(state_history, dtype=np.float32)
        actions = np.array(action_history, dtype=np.float32)
        rtgs = np.array(rtg_history, dtype=np.float32).reshape(-1, 1)
        timesteps = np.array(timestep_history, dtype=np.int64)

        current_len = min(t + 1, MAX_LENGTH)
        mask = np.concatenate(
            [np.zeros(MAX_LENGTH - current_len), np.ones(current_len)], dtype=np.float32
        )

        states_tensor = (
            torch.tensor(states, dtype=torch.float32).unsqueeze(0).to(device)
        )
        actions_tensor = (
            torch.tensor(actions, dtype=torch.float32).unsqueeze(0).to(device)
        )
        rtgs_tensor = torch.tensor(rtgs, dtype=torch.float32).unsqueeze(0).to(device)
        timesteps_tensor = (
            torch.tensor(timesteps, dtype=torch.long).unsqueeze(0).to(device)
        )
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            model_outputs = model(
                states=states_tensor,
                actions=actions_tensor,
                returns_to_go=rtgs_tensor,
                timesteps=timesteps_tensor,
                attention_mask=mask_tensor,
                return_dict=True,
            )
            action_preds = model_outputs["action_preds"]
            action = action_preds[0, -1].cpu().numpy()

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        action_history.append(action.astype(np.float32))
        state_history.append(obs.astype(np.float32))
        rtg_history.append(rtg_history[-1] - reward)
        timestep_history.append(t + 1)

        print(
            f"Current Reward: {total_reward:.2f} | Desired Reward Left: {rtg_history[-1]:.2f} | Desired Horizon Left: {d_h - (t + 1):.2f}"
        )
        env.render()
        time.sleep(time_interval)

        if terminated or truncated:
            break


def run_low_reward_test(
    env: gym.Env,
    model,
    d_r: float,
    d_h: float,
    model_type: str,
    device: str,
    time_interval: float = 0.05,
):
    """Dispatches the low reward test to the appropriate model-specific function."""
    if model_type == "NeuralNet":
        _run_nn_low_reward_test(env, model, d_r, d_h, device, time_interval)
    elif model_type == "DecisionTransformer":
        _run_dt_low_reward_test(env, model, d_r, d_h, device, time_interval)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    env.close()


if __name__ == "__main__":
    evaluate_dt = False  # Set to True to evaluate DT, False for NN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("Ant-v5", render_mode="human")
    d_r = 0.0
    d_h = 1000.0
    time_interval = 0.02  # Set the desired time interval for rendering
    model = None
    model_type = "NeuralNet"

    if not evaluate_dt:
        model = load_nn_model_for_eval(
            INPUT_SIZE, 256, OUTPUT_SIZE, NN_MODEL_PATH, device
        )
        model_type = "NeuralNet"
    else:
        model = load_dt_model_for_eval(
            STATE_DIM, OUTPUT_SIZE, MAX_LENGTH, DT_MODEL_PATH, device
        )
        model_type = "DecisionTransformer"

    if model is not None:
        run_low_reward_test(env, model, d_r, d_h, model_type, device, time_interval)
