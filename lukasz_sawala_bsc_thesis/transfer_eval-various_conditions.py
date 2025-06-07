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
    AntMazeBERTPretrainedMazeWrapper,
    AntMazeNNPretrainedMazeWrapper
)
from model_evaluation import (
    load_nn_model_for_eval, load_bert_mlp_model_for_eval,
    NN_MODEL_PATH, BERT_MLP_MODEL_PATH,
)
from utils import parse_arguments
from model_evaluation_ALL import plot_all_models_rewards
from transfer_eval_main import extract_goal_direction, antmaze_evaluate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



if __name__ == "__main__":
    args = parse_arguments(training=False)
    gym.register_envs(gymnasium_robotics)

    env = gym.make("AntMaze_MediumDense-v5")

    # Define d_r test values
    d_h = 1000.0
    num_episodes = args["episodes"]

    results = {
        "NeuralNet": {"avg_rewards": [], "sem": [], "percent_errors": []},
        "BERT_MLP": {"avg_rewards": [], "sem": [], "percent_errors": []},
    }
    
    # ======================= CONDITION 1: MODELS TAKEN FROM ANT WITH NO EXTRA TRAINING =========================
    # d_r_options = [i * 50 for i in range(args["d_r_array_length"])]

    # nn_base, _ = load_nn_model_for_eval(107, 256, 8, NN_MODEL_PATH, DEVICE)
    # nn_model = AntNNPretrainedMazePolicy(nn_base, action_dim=8).to(DEVICE)

    # bert_base = load_bert_mlp_model_for_eval(BERT_MLP_MODEL_PATH, DEVICE)
    # bert_mlp_model = AntBERTPretrainedMazePolicy(*bert_base, init_head=True).to(DEVICE)
    # use_goal = False  # No goal direction in this condition
    # state_dim = 105  # State dimension for AntMaze environment

    # models = {
    #     "NeuralNet": (nn_model, state_dim, use_goal),
    #     "BERT_MLP": (bert_mlp_model, state_dim, use_goal),
    # }
    # save_path = "condition1-2models.png"
    # =============================================================================================================

    # =========================== CONDITION 2: MODELS TAKEN FROM ANT WITH MAZE FINETUNING =========================
    d_r_options = [i * 50 for i in range(args["d_r_array_length"])]

    if "finetune" not in NN_MODEL_PATH or "finetune" not in BERT_MLP_MODEL_PATH:
        raise ValueError("Model paths must point to finetuned models for this condition.")
    nn_base, actionhead = load_nn_model_for_eval(107, 256, 8, NN_MODEL_PATH, DEVICE)
    nn_model = AntNNPretrainedMazePolicy(nn_base, action_dim=8, adjusted_head=actionhead).to(DEVICE)

    bert_base = load_bert_mlp_model_for_eval(BERT_MLP_MODEL_PATH, DEVICE, antmaze_pretrained=True)
    bert_model = AntBERTPretrainedMazePolicy(*bert_base[0:3], init_head=False,
                                             adjusted_head=bert_base[3], hidden_size=512).to(DEVICE)

    use_goal = True
    state_dim = 105  # State dimension for AntMaze environment
    models = {
        "NeuralNet": (nn_model, state_dim, use_goal),
        "BERT_MLP": (bert_model, state_dim, use_goal),
    }
    save_path = "condition2-2models.png"
    # =============================================================================================================

    # =========================== CONDITION 3: MODELS TRAINED ON ANTMAZE ==========================================
    # d_r_options = [i * 50 for i in range(args["d_r_array_length"])]
    #
    # model_components = load_antmaze_bertmlp_model_for_eval(ANTMAZE_BERT_PATH, DEVICE)
    # bert_model = AntMazeBERTPretrainedMazeWrapper(*model_components).to(DEVICE)
    #
    # nn_model_base = load_antmaze_nn_model_for_eval(ANTMAZE_NN_PATH, DEVICE)
    # nn_model = AntMazeNNPretrainedMazeWrapper(nn_model_base).to(DEVICE)
    #
    # use_goal = True
    # state_dim = 27  # Reduced state space due to dataset mismatch
    # models = {
    #     "ANTMAZE_BERT_MLP": (bert_model, state_dim, use_goal),
    #     "ANTMAZE_NN": (nn_model, state_dim, use_goal),
    # }
    # save_path = "condition3-2models.png"
    # =============================================================================================================

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
    plot_all_models_rewards(results, d_r_options, save_path=f"eval-{save_path}",)