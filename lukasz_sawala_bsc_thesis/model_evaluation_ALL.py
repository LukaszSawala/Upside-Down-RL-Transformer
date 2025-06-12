import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem
# from zeus.monitor import ZeusMonitor
from utils import parse_arguments
from model_evaluation import (
    evaluate_get_rewards, load_bert_mlp_model_for_eval,
    load_bert_udrl_model_for_eval, load_nn_model_for_eval, load_dt_model_for_eval,
    NN_MODEL_PATH, DT_MODEL_PATH, BERT_UDRL_MODEL_PATH, BERT_MLP_MODEL_PATH
)


OUTPUT_SIZE = 8
PLOT_TITLE = "Obtained Reward vs. D_r"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Best loss recorded: CLS, batch 8, lr 5e-5, epoch 1: 0.012
supaugmented; loss 0.02
"""

MAX_LENGTH = 60
INPUT_SIZE = 105 + 2  # s_t + d_r and d_t
STATE_DIM = INPUT_SIZE - 2  # used for the DT


def plot_all_models_rewards(
    results: dict,
    d_r_values: list,
    save_path="!!average_rewards_all_models.png",
):
    """
    Plots average rewards for all models with standard error bands.

    Args:
        results: dict of model_name -> {'avg_rewards': list, 'sem': list}
        d_r_values: list of d_r values
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 8))

    custom_palette = [
        "#F5C242",  # warm yellow
        "#F26464",  # coral red
        "#76C893",  # minty green
        "#8E7CC3",   # rich lavender
        "#465CB2",  # midnight blue
        "#FF811A",  # soft orange
    ]
    palette = sns.color_palette(custom_palette)

    for i, (model_name, data) in enumerate(results.items()):
        avg_rewards = np.array(data["avg_rewards"])
        sem_vals = np.array(data["sem"])
        if model_name == "BERT_UDRL":
            model_name = "UDRLt"
        if model_name == "BERT_MLP":
            model_name = "UDRLt_MLP"
        plt.plot(d_r_values, avg_rewards, label=model_name, color=palette[i], marker="o")
        plt.fill_between(
            d_r_values,
            avg_rewards - sem_vals,
            avg_rewards + sem_vals,
            alpha=0.2,
            color=palette[i],
        )

    plt.plot(d_r_values, d_r_values, linestyle="dotted", color="gray", label="Ideal (y=x)")
    plt.xlabel("Desired Reward (d_r)", fontsize=14)
    plt.ylabel("Average Episodic Reward", fontsize=14)
    plt.title(PLOT_TITLE, fontsize=16, fontweight="bold")
    plt.legend()
    plt.ylim(0, max(max([max(data["avg_rewards"]) for data in results.values()]), max(d_r_values)) * 1.1)
    plt.savefig(save_path)
    print(f"Saved combined plot to {save_path}")
    plt.close()


if __name__ == "__main__":
    args = parse_arguments(training=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("starting evaluation for args:", args, "device:", device)

    hidden_size = 256
    nn_model = load_nn_model_for_eval(INPUT_SIZE, hidden_size, OUTPUT_SIZE, NN_MODEL_PATH, device)
    dt_model = load_dt_model_for_eval(STATE_DIM, OUTPUT_SIZE, MAX_LENGTH, DT_MODEL_PATH, device)
    bert_model = load_bert_udrl_model_for_eval(105, OUTPUT_SIZE, BERT_UDRL_MODEL_PATH, device)
    model_bert, state_encoder, mlp = load_bert_mlp_model_for_eval(BERT_MLP_MODEL_PATH, device)
    bert_mlp_model = (model_bert, state_encoder, mlp)

    d_h = 1000.0
    d_r_options = [i * 100 for i in range(args["d_r_array_length"])]
    num_episodes = args["episodes"]

    env = gym.make("Ant-v5")

    results = {
        "NeuralNet": {"avg_rewards": [], "sem": [], "percent_errors": []},
        "DecisionTransformer": {"avg_rewards": [], "sem": [], "percent_errors": []},
        "BERT_UDRL": {"avg_rewards": [], "sem": [], "percent_errors": []},
        "BERT_MLP": {"avg_rewards": [], "sem": [], "percent_errors": []},
    }

    models_dict = {
        "NeuralNet": nn_model,
        "DecisionTransformer": dt_model,
        "BERT_UDRL": bert_model,
        "BERT_MLP": bert_mlp_model,
    }

    for d_r in d_r_options:
        print("=" * 50)
        print(f"Evaluating d_r: {d_r}")

        for model_name, model in models_dict.items():
            print(f"Evaluating model: {model_name}")

            _, episodic_rewards = evaluate_get_rewards(
                env,
                model,
                d_h,
                d_r,
                num_episodes=num_episodes,
                model_type=model_name,
                device=device,
            )

            avg = np.mean(episodic_rewards)
            sem_val = sem(episodic_rewards)
            if d_r > 0:
                percent_error = abs(avg - d_r) / d_r
            else:
                percent_error = 0

            results[model_name]["avg_rewards"].append(avg)
            results[model_name]["sem"].append(sem_val)
            results[model_name]["percent_errors"].append(percent_error)

    env.close()

    print("\n" + "=" * 60)
    print("Final Average Percentage Errors per Model:")
    for model_name, data in results.items():
        mean_error = np.mean(data["percent_errors"]) * 100
        print(f"{model_name}: {mean_error:.2f}%")
    plot_all_models_rewards(results, d_r_options)
