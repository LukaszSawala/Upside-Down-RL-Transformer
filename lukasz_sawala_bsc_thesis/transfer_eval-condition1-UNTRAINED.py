import gymnasium as gym
import torch
import numpy as np
import time
import gymnasium_robotics
from scipy.stats import sem
from transformers import AutoConfig, AutoModel
import torch.nn as nn
from models import (
    AntNNPretrainedMazePolicy,
    AntBERTPretrainedMazePolicy,
)
from model_evaluation import (
    load_nn_model_for_eval, load_bert_mlp_model_for_eval,
    NN_MODEL_PATH, BERT_MLP_MODEL_PATH,
)
from utils import parse_arguments
from model_evaluation_ALL import plot_all_models_rewards
from transferrability_eval import extract_goal_direction

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def antmaze_evaluate(
    env, model, episodes=10, time_interval=0.05, d_r=5.0, d_h=1000.0, state_dim=105, use_goal=False
):
    best_distances = []
    obtained_returns = []
    for episode in range(episodes):
        print(f"Episode: {episode}")
        obs = env.reset()[0]  # extract the values from the wrapped array
        done = False
        d_h_copy, d_r_copy = d_h, d_r
        best_distance = 1000
        total_reward = 0
        while not done and d_h_copy > 0:
            goal_vec = extract_goal_direction(obs)
            distance = np.linalg.norm(goal_vec)
            if distance < best_distance:
                best_distance = distance
            obs = obs["observation"][:state_dim]  # exytract the values from the wrapped array

            with torch.no_grad():
                action_tensor = model(obs, d_r_copy, d_h_copy, goal_vec, DEVICE, use_goal=use_goal)
            action = action_tensor.squeeze(0).cpu().numpy()

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            d_r_copy -= reward
            d_h_copy -= 1
            done = terminated or truncated

            #env.render()
            #time.sleep(time_interval)
        obtained_returns.append(total_reward)
        best_distances.append(best_distance)
    print("minimum return:", min(obtained_returns), "maximum return:", max(obtained_returns))
    return obtained_returns, best_distances


if __name__ == "__main__":
    args = parse_arguments(training=False)
    gym.register_envs(gymnasium_robotics)

    env = gym.make("AntMaze_MediumDense-v5")

    # Define d_r test values
    d_r_options = [i * 10 for i in range(args["d_r_array_length"])]
    d_h = 1000.0
    num_episodes = args["episodes"]

    results = {
        "NeuralNet": {"avg_rewards": [], "sem": [], "percent_errors": []},
        "BERT_MLP": {"avg_rewards": [], "sem": [], "percent_errors": []},
    }

    # Load and wrap models
    nn_base = load_nn_model_for_eval(107, 256, 8, NN_MODEL_PATH, DEVICE)
    nn_model = AntNNPretrainedMazePolicy(nn_base, action_dim=8).to(DEVICE)

    bert_base = load_bert_mlp_model_for_eval(BERT_MLP_MODEL_PATH, DEVICE)
    bert_mlp_model = AntBERTPretrainedMazePolicy(*bert_base, init_head=True).to(DEVICE)

    models = {
        "NeuralNet": (nn_model, 105, False),
        "BERT_MLP": (bert_mlp_model, 105, False),
    }

    for d_r in d_r_options:
        print("=" * 50)
        print(f"Evaluating d_r: {d_r}")
        for name, (model, state_dim, use_goal) in models.items():
            print(f"Evaluating model: {name}")
            returns, _ = antmaze_evaluate(env, model, num_episodes, d_r=d_r,
                                          d_h=d_h, state_dim=state_dim, use_goal=use_goal)
            avg = np.mean(returns)
            se = sem(returns)
            error = abs(avg - d_r) / d_r if d_r > 0 else 0
            results[name]["avg_rewards"].append(avg)
            results[name]["sem"].append(se)
            results[name]["percent_errors"].append(error)

    env.close()

    print("\n" + "=" * 60)
    print("Final Average Percentage Errors per Model:")
    for model_name, data in results.items():
        mean_error = np.mean(data["percent_errors"]) * 100
        print(f"{model_name}: {mean_error:.2f}%")

    # Final multi-model plot
    plot_all_models_rewards(results, d_r_options, save_path="condition1-2models.png",)