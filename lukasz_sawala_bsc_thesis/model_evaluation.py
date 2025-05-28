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
from utils import parse_arguments, print_available_antmaze_envs
from models import NeuralNet, ActionHead, LargeActionHead, ScalarEncoder, HugeNeuralNet, AntMazeActionHead


OUTPUT_SIZE = 8
#NN_MODEL_PATH = "../models/best_nn_grid.pth"
NN_MODEL_PATH = "finetunedNN-512.pth"  # for evaluation
DT_MODEL_PATH = "../models/best_DT_grid.pth"
BERT_UDRL_MODEL_PATH = "../models/bert_tiny.pth"
#BERT_MLP_MODEL_PATH = "../models/mlpbert_t_hugemlp.pth"  # for finetuning
BERT_MLP_MODEL_PATH = "finetunedbroski-512.pth"    # for evaluation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Best loss recorded: CLS, batch 8, lr 5e-5, epoch 1: 0.012
supaugmented; loss 0.02
"""

MAX_LENGTH = 60
INPUT_SIZE = 105 + 2  # s_t + d_r and d_t
STATE_DIM = INPUT_SIZE - 2  # used for the DT


def load_nn_model_for_eval(input_size: int, hidden_size: int,
                           output_size: int, checkpoint_path: str,
                           device: str) -> NeuralNet:
    """Loads a Neural Network model for evaluation."""
    model = NeuralNet(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size
    )
    action_head = None
    if "finetuned" in checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["nn"])
        action_head = AntMazeActionHead(hidden_size=512, act_dim=8).to(device)
        action_head.load_state_dict(checkpoint["action_head"])
        action_head.eval()
    model.to(device)
    model.eval()
    return model, action_head 


def load_dt_model_for_eval(state_dim: int, act_dim: int,
                           max_length: int, checkpoint_path: str,
                           device: str) -> DecisionTransformerModel:
    """Loads a Decision Transformer model for evaluation."""
    config = DecisionTransformerConfig(
        state_dim=state_dim, act_dim=act_dim, max_length=max_length
    )
    model = DecisionTransformerModel(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_bert_udrl_model_for_eval(state_dim: int, act_dim: int,
                                  checkpoint_path: str, device: str) -> tuple:
    """
    Loads the bert-based UDRLt model components for evaluation.
    Args:
        state_dim (int): The dimension of the state.
        act_dim (int): The dimension of the action.
        checkpoint_path (str): The path to the checkpoint file.
        device (str): The device to load the model on.
    """
    config = AutoConfig.from_pretrained("prajjwal1/bert-tiny")
    config.vocab_size = 1  # dummy since we're using inputs_embeds
    config.max_position_embeddings = 3
    model_bert = AutoModel.from_config(config).to(device)
    d_r_encoder = nn.Linear(1, config.hidden_size).to(device)
    d_h_encoder = nn.Linear(1, config.hidden_size).to(device)
    #d_r_encoder = ScalarEncoder(config.hidden_size).to(device)
    #d_h_encoder = ScalarEncoder(config.hidden_size).to(device)
    state_encoder = nn.Linear(state_dim, config.hidden_size).to(device)
    head = nn.Linear(config.hidden_size, act_dim).to(device)
    #head = ActionHead(config.hidden_size, act_dim).to(device)
    #head = LargeActionHead(config.hidden_size, act_dim).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_bert.load_state_dict(checkpoint["bert"])
    d_r_encoder.load_state_dict(checkpoint["d_r"])
    d_h_encoder.load_state_dict(checkpoint["d_t"])
    state_encoder.load_state_dict(checkpoint["state"])
    head.load_state_dict(checkpoint["head"])

    model_bert.eval()
    d_r_encoder.eval()
    d_h_encoder.eval()
    state_encoder.eval()
    head.eval()

    return model_bert, d_r_encoder, d_h_encoder, state_encoder, head


def load_bert_mlp_model_for_eval(checkpoint_path: str, device: str, freeze: bool = False, antmaze_pretrained: bool = False):
    """
    Loads the bert-based MLP model components for evaluation.
    Args:
        checkpoint_path (str): The path to the checkpoint file.
        device (str): The device to load the model on.
        freeze (bool): Whether to freeze the BERT model.
        antmaze_pretrained (bool): Whether to return the fine-tuned AntMaze action head.
    """
    # Load BERT config
    config = AutoConfig.from_pretrained("prajjwal1/bert-tiny")
    config.vocab_size = 1
    config.max_position_embeddings = 1

    # Initialize components
    model_bert = AutoModel.from_config(config).to(device)
    state_encoder = nn.Linear(105, config.hidden_size).to(device)
    #mlp = NeuralNet(input_size=config.hidden_size + 2, hidden_size=256, output_size=8).to(device)
    mlp = HugeNeuralNet(input_size=config.hidden_size + 2, hidden_size=256, output_size=8).to(device)

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_bert.load_state_dict(checkpoint["bert"])
    state_encoder.load_state_dict(checkpoint["state"])
    mlp.load_state_dict(checkpoint["mlp"])

    model_bert.eval()
    state_encoder.eval()
    mlp.eval()
    if freeze:
        print("Freezing base model parameters")
        for param in model_bert.parameters():
            param.requires_grad = False
        for param in state_encoder.parameters():
            param.requires_grad = False
        for param in mlp.parameters():
            param.requires_grad = False

    if antmaze_pretrained:
        action_head = AntMazeActionHead(hidden_size=512, act_dim=8).to(DEVICE)
        action_head.load_state_dict(checkpoint["action_head"])
        action_head.eval()
        return model_bert, state_encoder, mlp, action_head
    return model_bert, state_encoder, mlp


def evaluate_get_rewards(env: gym.Env, model, d_h: float,
                         d_r: float, num_episodes: int = 1,
                         max_episode_length: int = 1000,
                         model_type: str = "NeuralNet",
                         device: str = "cpu") -> tuple:
    """
    Evaluate the performance of the model on the given environment.
    Args:
        env (gym.Env): The environment to evaluate on.
        model: The model to evaluate.
        d_h (float): The horizon to go.
        d_r (float): The reward to go.
        num_episodes (int): The number of episodes to evaluate.
        max_episode_length (int): The maximum length of an episode.
        model_type (str): The type of model to evaluate.
        device (str): The device to evaluate on.
    Returns:
        tuple: A tuple containing the average reward and a list of episodic rewards.
    """
    if model_type == "NeuralNet":
        return _evaluate_neural_net(
            env, model, d_h, d_r, num_episodes, max_episode_length
        )
    elif model_type == "DecisionTransformer":
        return _evaluate_decision_transformer(
            env, model, d_r, num_episodes, max_episode_length, device
        )
    elif model_type == "BERT_UDRL":
        return _evaluate_bert_udrl(
            env, *model, d_r, d_h, num_episodes, max_episode_length, device
        )
    elif model_type == "BERT_MLP":
        return _evaluate_bert_mlp(
            env, *model, d_r, d_h, num_episodes, max_episode_length, device
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def _evaluate_neural_net(env: gym.Env, model, d_h: float,
                         d_r: float, num_episodes: int,
                         max_episode_length: int) -> tuple:
    """
    Evaluate the performance of the Neural Network model.
    """
    episodic_rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        d_r_copy, d_h_copy = d_r, d_h
        total_reward = 0
        for _ in range(max_episode_length):
            obs_input = (
                torch.tensor(
                    np.concatenate((obs, [d_r_copy, d_h_copy])), dtype=torch.float32
                )
                .unsqueeze(0)
                .to(DEVICE)
            )
            with torch.no_grad():
                action = model(obs_input).squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            d_r_copy -= reward
            d_h_copy -= 1
            if terminated or truncated:
                break
        episodic_rewards.append(total_reward)
    print(
        "max-min reward for this dr:", max(episodic_rewards), "-", min(episodic_rewards)
    )
    return np.mean(episodic_rewards), episodic_rewards


def _evaluate_decision_transformer(env: gym.Env, model, d_r: float,
                                   num_episodes: int, max_episode_length: int,
                                   device: str = "cpu") -> tuple:
    """
    Evaluate the performance of the Decision Transformer model.
    """
    episodic_rewards = []
    act_dim = model.config.act_dim
    state_dim = model.config.state_dim

    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        target_return = float(d_r)

        state_history = deque(
            [np.zeros(state_dim, dtype=np.float32)] * MAX_LENGTH, maxlen=MAX_LENGTH
        )
        action_history = deque(
            [np.ones(act_dim, dtype=np.float32) * -10.0] * MAX_LENGTH, maxlen=MAX_LENGTH
        )
        rtg_history = deque([0.0] * MAX_LENGTH, maxlen=MAX_LENGTH)
        timestep_history = deque([0] * MAX_LENGTH, maxlen=MAX_LENGTH)

        state_history.append(obs.astype(np.float32))
        rtg_history.append(target_return)

        for t in range(max_episode_length):
            states = np.array(state_history, dtype=np.float32)
            actions = np.array(action_history, dtype=np.float32)
            rtgs = np.array(rtg_history, dtype=np.float32).reshape(-1, 1)
            timesteps = np.array(timestep_history, dtype=np.int64)

            current_len = min(t + 1, MAX_LENGTH)
            mask = np.concatenate(
                [np.zeros(MAX_LENGTH - current_len), np.ones(current_len)],
                dtype=np.float32,
            )

            states_tensor = (
                torch.tensor(states, dtype=torch.float32).unsqueeze(0).to(device)
            )
            actions_tensor = (
                torch.tensor(actions, dtype=torch.float32).unsqueeze(0).to(device)
            )
            rtgs_tensor = (
                torch.tensor(rtgs, dtype=torch.float32).unsqueeze(0).to(device)
            )
            timesteps_tensor = (
                torch.tensor(timesteps, dtype=torch.long).unsqueeze(0).to(device)
            )
            mask_tensor = (
                torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)
            )
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
            current_rtg = rtg_history[-1]
            rtg_history.append(current_rtg - reward)
            timestep_history.append(t + 1)

            if terminated or truncated:
                break

        episodic_rewards.append(total_reward)
    print(
        "max-min reward for this dr (DT):",
        max(episodic_rewards),
        "-",
        min(episodic_rewards),
    )
    return np.mean(episodic_rewards), episodic_rewards


def _evaluate_bert_udrl(env: gym.Env, model_bert, d_r_encoder,
                        d_h_encoder, state_encoder, head, d_r: float,
                        d_h: float, num_episodes: int, max_episode_length: int,
                        device: str) -> tuple:
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
                sequence = torch.cat([encoded_r, encoded_h, encoded_s], dim=1)
                bert_out = model_bert(inputs_embeds=sequence).last_hidden_state
                action = head(bert_out[:, -1]).squeeze(0).cpu().numpy()

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


def _evaluate_bert_mlp(env: gym.Env, model_bert, state_encoder, head,
                       d_r: float, d_h: float, num_episodes: int,
                       max_episode_length: int, device: str) -> tuple:
    """
    Evaluate scalar-concat model: uses encoded state + scalar d_r and d_h.
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
                s_encoded = state_encoder(obs_tensor).unsqueeze(1)
                bert_out = model_bert(inputs_embeds=s_encoded).last_hidden_state[:, 0]
                mlp_input = torch.cat([bert_out, dr_tensor, dh_tensor], dim=1)
                action = head(mlp_input).squeeze(0).cpu().numpy()

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            d_r_copy -= reward
            d_h_copy -= 1
            if terminated or truncated:
                break

        episodic_rewards.append(total_reward)

    print(
        "max-min reward for this dr (BERT-MLP):",
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
    max_y: float = 5500,
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
    plt.ylim(0, max_y)
    print("max_y:", max_y, "ticks every:", max_y / 11)
    plt.yticks(np.arange(0, max_y, max_y / 11))
    sns.despine()
    plt.savefig(save_path)
    print(f"Average rewards plot saved in {save_path}")


if __name__ == "__main__":
    args = parse_arguments(training=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("starting evaluation for args:", args, "device:", device)
    model = None
    if args["model_type"] == "NeuralNet":
        hidden_size = 256
        model = load_nn_model_for_eval(
            INPUT_SIZE, hidden_size, OUTPUT_SIZE, NN_MODEL_PATH, device
        )
    elif args["model_type"] == "DecisionTransformer":
        model = load_dt_model_for_eval(
            STATE_DIM, OUTPUT_SIZE, MAX_LENGTH, DT_MODEL_PATH, device
        )
    elif args["model_type"] == "BERT_UDRL":
        model = load_bert_udrl_model_for_eval(
            105, OUTPUT_SIZE, BERT_UDRL_MODEL_PATH, device
        )
    elif args["model_type"] == "BERT_MLP":
        model_bert, state_encoder, mlp = load_bert_mlp_model_for_eval(
            BERT_MLP_MODEL_PATH, device
        )
        model = (model_bert, state_encoder, mlp)
    else:
        raise ValueError(f"Unsupported model_type: {args['model_type']}")

    d_h = 1000.0
    d_r_options = [i * 100 for i in range(args["d_r_array_length"])]
    num_episodes = args["episodes"]

    env = gym.make("Ant-v5")  # render mode 'human' for visualization
    average_rewards = []
    sem_values = []
    error_percentages = []
    # monitor = ZeusMonitor(gpu_indices=[0] if device.type == 'cuda' else [], cpu_indices=[0, 1])
    # monitor.begin_window(f"evaluation_{args['model_type']}")

    for d_r in d_r_options:
        drcopy = d_r
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

        if drcopy > 0:
            percentage_error_dr = abs(drcopy - np.mean(episodic_rewards)) / drcopy
            error_percentages.append(percentage_error_dr)
    # mes = monitor.end_window(f"evaluation_{args['model_type']}")
    # print(f"Training grid search took {mes.time} s and consumed {mes.total_energy} J.")

    save_path = f"average_rewards_plot_{args['model_type']}.png"
    plot_average_rewards(average_rewards, sem_values, d_r_options, save_path=save_path)
    print("Average error:", np.mean(error_percentages))
    env.close()
