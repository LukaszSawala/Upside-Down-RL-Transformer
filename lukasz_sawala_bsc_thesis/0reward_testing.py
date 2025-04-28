from models import NeuralNet
import torch
import gymnasium as gym
import numpy as np
import time


INPUT_SIZE = 105 + 2  # s_t + d_r and d_t
OUTPUT_SIZE = 8
NN_MODEL_PATH = "../models/best_nn_grid.pth"


def load_nn_model_for_eval(
    input_size: int,
    hidden_size: int,
    output_size: int,
    checkpoint_path: str,
) -> NeuralNet:
    """
    Loads a Neural Network model from a given checkpoint path for evaluation.
    """
    model = NeuralNet(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size
    )
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model


if __name__ == "__main__":
    env = gym.make("Ant-v5", render_mode="human")
    model = load_nn_model_for_eval(INPUT_SIZE, 256, OUTPUT_SIZE, NN_MODEL_PATH)
    d_r = 1000.0
    d_h = 1000.0
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    while not done or d_h > 0:
        obs_input = torch.tensor(
            np.concatenate((obs, [d_r, d_h])), dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():
            action = model(obs_input).squeeze(0).numpy()

        # Take a step in the environment
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        d_r -= reward  # Update the remaining desired reward
        d_h -= 1       # Update the remaining desired horizon

        # Print the current reward and progress
        print(f"Current Reward: {total_reward:.2f} | Desired Reward Left: {d_r:.2f} | Desired Horizon Left: {d_h:.2f}")

        env.render()
        time.sleep(0.1)

        if terminated or truncated:
            done = True

    env.close()
