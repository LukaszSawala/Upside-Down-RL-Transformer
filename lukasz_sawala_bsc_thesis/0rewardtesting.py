import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch

from model_evaluation import (
    load_dt_model_for_eval,
    load_nn_model_for_eval,
    load_bert_udrl_model_for_eval,
)


INPUT_SIZE = 105 + 2  # s_t + d_r and d_h
OUTPUT_SIZE = 8
NN_MODEL_PATH = "../models/best_nn_grid.pth"
DT_MODEL_PATH = "../models/best_DT_grid.pth"
UDRLT_MODEL_PATH = "../models/best_bert_udrl_rn.pth"
MAX_LENGTH = 60
STATE_DIM = INPUT_SIZE - 2


def _run_nn_low_reward_test(env, model, d_r, d_h, device, time_interval):
    """
    Evaluate the performance of the Neural Network model in the low reward conditions.
    """
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    low_reward_array = []
    first = None

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
        if d_r < 10:
            if first is None:
                print("NOW!" * 50)
                first = 1000 - d_h
            low_reward_array.append(reward)

        print(
            f"Current Reward: {total_reward:.2f} | Desired Reward Left: {d_r:.2f} | Desired Horizon Left: {d_h:.2f}"
        )
        env.render()
        time.sleep(time_interval)

        if terminated or truncated:
            done = True

    print(
        f"average reward obtained in the low reward conditions: {np.mean(low_reward_array)} started at {first}"
    )


def _run_dt_low_reward_test(env, model, d_r, d_h, device, time_interval):
    """
    Evaluate the performance of the Decision Transformer model in the low reward conditions.
    """
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

        with torch.no_grad():
            model_outputs = model(
                states=torch.tensor(states).unsqueeze(0).to(device),
                actions=torch.tensor(actions).unsqueeze(0).to(device),
                returns_to_go=torch.tensor(rtgs).unsqueeze(0).to(device),
                timesteps=torch.tensor(timesteps).unsqueeze(0).to(device),
                attention_mask=torch.tensor(mask).unsqueeze(0).to(device),
                return_dict=True,
            )
            action = model_outputs["action_preds"][0, -1].cpu().numpy()

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


def _run_udrlt_low_reward_test(
    env, model_bert, d_r_enc, d_h_enc, state_enc, head, d_r, d_h, device, time_interval
):
    """
    Evaluate the performance of the UDRLt model in the low reward conditions.
    """
    obs, _ = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        state_embed = state_enc(
            torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        )
        d_r_embed = d_r_enc(torch.tensor([[d_r]], dtype=torch.float32).to(device))
        d_h_embed = d_h_enc(torch.tensor([[d_h]], dtype=torch.float32).to(device))

        input_embeds = torch.stack(
            [d_r_embed.squeeze(1), d_h_embed.squeeze(1), state_embed.squeeze(1)], dim=1
        )

        with torch.no_grad():
            output = model_bert(inputs_embeds=input_embeds)
            action = head(output.last_hidden_state[:, -1]).squeeze(0).cpu().numpy()

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        d_r -= reward
        d_h -= 1

        print(
            f"Current Reward: {total_reward:.2f} | Desired Reward Left: {d_r:.2f} | Desired Horizon Left: {d_h:.2f}"
        )
        env.render()
        time.sleep(time_interval)

        if terminated or truncated or d_h <= 0:
            done = True


def run_low_reward_test(env, model, d_r, d_h, model_type, device, time_interval=0.05):
    """Evaluate the performance of the model in the low reward conditions."""
    if model_type == "NeuralNet":
        _run_nn_low_reward_test(env, model, d_r, d_h, device, time_interval)
    elif model_type == "DecisionTransformer":
        _run_dt_low_reward_test(env, model, d_r, d_h, device, time_interval)
    elif model_type == "UDRLt":
        model_bert, d_r_enc, d_h_enc, state_enc, head = model
        _run_udrlt_low_reward_test(
            env,
            model_bert,
            d_r_enc,
            d_h_enc,
            state_enc,
            head,
            d_r,
            d_h,
            device,
            time_interval,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    env.close()


if __name__ == "__main__":
    model_choice = (
        "NeuralNet"  # Change to "NeuralNet", "DecisionTransformer", or "UDRLt"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("Ant-v5", render_mode="human")
    d_r = 1000.0
    d_h = 1000.0
    time_interval = 0.05

    if model_choice == "NeuralNet":
        model = load_nn_model_for_eval(
            INPUT_SIZE, 256, OUTPUT_SIZE, NN_MODEL_PATH, device
        )
    elif model_choice == "DecisionTransformer":
        model = load_dt_model_for_eval(
            STATE_DIM, OUTPUT_SIZE, MAX_LENGTH, DT_MODEL_PATH, device
        )
    elif model_choice == "UDRLt":
        model = load_bert_udrl_model_for_eval(
            STATE_DIM, OUTPUT_SIZE, UDRLT_MODEL_PATH, device
        )
    else:
        raise ValueError(f"Invalid model_choice: {model_choice}")

    run_low_reward_test(env, model, d_r, d_h, model_choice, device, time_interval)
