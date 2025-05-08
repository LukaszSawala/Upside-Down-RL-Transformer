import torch
import gymnasium as gym
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem
from transformers import (
    DecisionTransformerModel,
    DecisionTransformerConfig,
    AutoModel,
    AutoConfig,
)
from collections import deque
# from zeus.monitor import ZeusMonitor 
from utils import parse_arguments
from models import NeuralNet, ActionHead



OUTPUT_SIZE = 8
NN_MODEL_PATH = "../models/best_nn_grid.pth"
DT_MODEL_PATH = "../models/best_DT_grid.pth"
BERT_UDRL_MODEL_PATH = "b_CLS_medium.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Best loss recorded: CLS, batch 8, lr 5e-5, epoch 1: 0.012
"""

MAX_LENGTH = 60
INPUT_SIZE = 105 + 2  # s_t + d_r and d_t
STATE_DIM = INPUT_SIZE - 2  # used for the DT


def load_bert_udrl_model_for_eval(state_dim: int, act_dim: int,
                                  checkpoint_path: str, device: str) -> tuple:
    """Loads the BERT UDRL model components for evaluation."""
    config = AutoConfig.from_pretrained("prajjwal1/bert-medium")
    config.vocab_size = 1  # dummy since we're using inputs_embeds
    config.max_position_embeddings = 4
    model_bert = AutoModel.from_config(config).to(device)
    d_r_encoder = nn.Linear(1, config.hidden_size).to(device)
    d_h_encoder = nn.Linear(1, config.hidden_size).to(device)
    state_encoder = nn.Linear(state_dim, config.hidden_size).to(device)
    head = nn.Linear(config.hidden_size, act_dim).to(device)
    # head = ActionHead(config.hidden_size, act_dim).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_bert.load_state_dict(checkpoint["bert"])
    d_r_encoder.load_state_dict(checkpoint["d_r"])
    d_h_encoder.load_state_dict(checkpoint["d_h"])
    state_encoder.load_state_dict(checkpoint["state"])
    head.load_state_dict(checkpoint["head"])
    cls_embedding = nn.Parameter(checkpoint['cls_embed'].to(device))


    model_bert.eval()
    d_r_encoder.eval()
    d_h_encoder.eval()
    state_encoder.eval()
    head.eval()

    return model_bert, d_r_encoder, d_h_encoder, state_encoder, head, cls_embedding


def evaluate_get_rewards(env: gym.Env, model, d_h: float,
                         d_r: float, num_episodes: int = 1,
                         max_episode_length: int = 1000,
                         model_type: str = "NeuralNet",
                         device: str = "cpu") -> tuple:
    """
    Evaluate the performance of the model on the given environment.
    """
    if model_type == "BERT_UDRL":
        return _evaluate_bert_udrl(
            env, *model, d_r, d_h, num_episodes, max_episode_length, device
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")





def _evaluate_bert_udrl(env: gym.Env, model_bert, d_r_encoder,
                        d_h_encoder, state_encoder, head, cls_embedding,
                        d_r: float, d_h: float, num_episodes: int,
                        max_episode_length: int, device: str) -> tuple:
    """
    Evaluate the performance of the BERT UDRL model.
    """
    episodic_rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        d_r_copy, d_h_copy = d_r, d_h
        total_reward = 0
        for _ in range(max_episode_length):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            dr_tensor = torch.tensor([d_r_copy], dtype=torch.float32).unsqueeze(0).to(device)
            dh_tensor = torch.tensor([d_h_copy], dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                encoded_r = d_r_encoder(dr_tensor).unsqueeze(1)  # reward to go
                encoded_h = d_h_encoder(dh_tensor).unsqueeze(1)  # horizon to go
                encoded_s = state_encoder(obs_tensor).unsqueeze(1)  # state
                batch_cls_embedding = cls_embedding.expand(obs_tensor.size(0), -1, -1)
                sequence = torch.cat([batch_cls_embedding, encoded_r, encoded_h, encoded_s], dim=1)
                bert_out = model_bert(inputs_embeds=sequence).last_hidden_state
                action = head(bert_out[:, 0]).squeeze(0).cpu().detach().numpy()


            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            d_r_copy -= reward
            d_h_copy -= 1
            if terminated or truncated:
                break
        episodic_rewards.append(total_reward)
    print(
        "max-min reward for this dr (BERT UDRL):",
        max(episodic_rewards),
        "-",
        min(episodic_rewards),
    )
    return np.mean(episodic_rewards), episodic_rewards


def plot_average_rewards(
    average_rewards: list,
    sem_values: list,
    d_r_values: list,
    title="Average Reward vs. d_r",
    save_path: str = "average_rewards_plot.png",
):
    """
    Plots the average rewards for different values of d_r with standard error bars.
    """
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.figure(figsize=(12, 7))
    sns.lineplot(
        x=d_r_values,
        y=average_rewards,
        linewidth=2.5,
        color="royalblue",
        marker="o",
        label="Average Reward",
    )
    plt.fill_between(
        d_r_values,
        np.array(average_rewards) - np.array(sem_values),
        np.array(average_rewards) + np.array(sem_values),
        color="royalblue",
        alpha=0.2,
    )
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
    model = None
    if args["model_type"] == "BERT_UDRL":
        model = load_bert_udrl_model_for_eval(
            105, OUTPUT_SIZE, BERT_UDRL_MODEL_PATH, device
        )
    else:
        raise ValueError(f"Unsupported model_type: {args['model_type']}")

    d_h = 1000.0
    d_r_options = [2000 + i * 100 for i in range(args["d_r_array_length"])]
    num_episodes = args["episodes"]

    env = gym.make("Ant-v5")  # render mode 'human' for visualization
    average_rewards = []
    sem_values = []

    # monitor = ZeusMonitor(gpu_indices=[0] if device.type == 'cuda' else [], cpu_indices=[0, 1])
    # monitor.begin_window(f"evaluation_{args['model_type']}")

    for d_r in d_r_options:
        print("=" * 50)
        print("Trying with d_r:", d_r)
        _, episodic_rewards = evaluate_get_rewards(
            env,
            model,
            d_h,
            d_r,
            num_episodes=num_episodes,
            model_type=args["model_type"],
            device=device,
        )
        average_rewards.append(np.mean(episodic_rewards))
        sem_values.append(sem(episodic_rewards))

    # mes = monitor.end_window(f"evaluation_{args['model_type']}")
    # print(f"Training grid search took {mes.time} s and consumed {mes.total_energy} J.")

    save_path = f"average_rewards_plot_{args['model_type']}.png"
    plot_average_rewards(average_rewards, sem_values, d_r_options, save_path=save_path)
    env.close()
