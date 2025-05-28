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
    HugeNeuralNet, NeuralNet10, NeuralNet12, NeuralNet16
)
from model_evaluation import (
    load_nn_model_for_eval, load_bert_mlp_model_for_eval,
    NN_MODEL_PATH, BERT_MLP_MODEL_PATH,
)
from utils import parse_arguments
from model_evaluation import plot_average_rewards, print_available_antmaze_envs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ANTMAZE_BERT_PATH = "antmaze_tiny-16_256.pth"


def load_antmaze_bertmlp_model_for_eval(checkpoint_path: str, device: str):
    config = AutoConfig.from_pretrained("prajjwal1/bert-tiny")
    config.vocab_size = 1
    config.max_position_embeddings = 1

    # Initialize components
    model_bert = AutoModel.from_config(config).to(device)
    state_encoder = nn.Linear(27, config.hidden_size).to(device)
    
    # hidden size + 4 for d_r, d_h and x y values of the goal vector
    # mlp = NeuralNet(input_size=config.hidden_size + 4, hidden_size=256, output_size=8).to(device)
    #mlp = NeuralNet10(input_size=config.hidden_size + 4, hidden_size=256, output_size=8).to(device)  
    #mlp = NeuralNet12(input_size=config.hidden_size + 4, hidden_size=128, output_size=8).to(device)  
    mlp = NeuralNet16(input_size=config.hidden_size + 4, hidden_size=512, output_size=8).to(device)  

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_bert.load_state_dict(checkpoint["bert"])
    state_encoder.load_state_dict(checkpoint["state"])
    mlp.load_state_dict(checkpoint["mlp"])

    # Set models to evaluation mode
    model_bert.eval()
    state_encoder.eval()
    mlp.eval()

    return model_bert, state_encoder, mlp


def extract_goal_direction(obs: dict) -> np.ndarray:
    """
    Calculate the direction vector from the current position to the goal position.

    Parameters:
        obs (dict): A dictionary containing 'desired_goal' and 'achieved_goal' keys,
                    where each key maps to a tuple of (x, y) coordinates.

    Returns:
        numpy.ndarray: A 2D numpy array representing the direction vector (dx, dy),
                       where dx and dy are the differences in the x and y coordinates
                       between the goal and the current position respectively.
    """
    goal_x, goal_y = obs['desired_goal']
    current_x, current_y = obs['achieved_goal']
    return np.array([goal_x - current_x, goal_y - current_y])


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
    # print_available_antmaze_envs() # check whether its compatible
    env = gym.make("AntMaze_MediumDense-v5")  # render mode human to see whats up

    # --- load models and wrap them to accept goal locations if necessary ------
    if args["model_type"] == "NeuralNet":
        nn_base, actionhead = load_nn_model_for_eval(107, 256, 8, NN_MODEL_PATH, DEVICE)
        model = AntNNPretrainedMazePolicy(nn_base, action_dim=8, adjusted_head=actionhead).to(DEVICE)
        use_goal = True if "finetuned" in NN_MODEL_PATH else False
        state_dim = 105
    elif args["model_type"] == "BERT_MLP":
        if "finetuned" in BERT_MLP_MODEL_PATH:
            bert_base = load_bert_mlp_model_for_eval(BERT_MLP_MODEL_PATH, DEVICE, antmaze_pretrained=True)
            model = AntBERTPretrainedMazePolicy(*bert_base[0:3], init_head=False, adjusted_head=bert_base[3], hidden_size=512).to(DEVICE)
            use_goal = True
        else:
            bert_base = load_bert_mlp_model_for_eval(BERT_MLP_MODEL_PATH, DEVICE)
            model = AntBERTPretrainedMazePolicy(*bert_base, init_head=True).to(DEVICE)
            use_goal = False
        state_dim = 105
    elif args["model_type"] == "ANTMAZE_BERT_MLP":
        model_components = load_antmaze_bertmlp_model_for_eval(ANTMAZE_BERT_PATH, DEVICE)
        model = AntMazeBERTPretrainedMazeWrapper(*model_components).to(DEVICE)
        state_dim = 27  # reduced state space due to dataset mismatch
        use_goal = True
    else:
        raise ValueError(f"Unsupported model_type: {args['model_type']}")

    d_h = 1000.0
    d_r_options = [i * 50 for i in range(args["d_r_array_length"])] # test those out
    if max(d_r_options) < 100:
        print("LOW REWARD TESTING")
    num_episodes = args["episodes"]
    average_rewards = []
    sem_values = []
    success_rates = []
    best_distances = []
    print("Evaluating AntMaze with model:", args["model_type"])
    for d_r in d_r_options:
        print("=" * 50)
        print("Trying with d_r:", d_r)
        returns, distances = antmaze_evaluate(
            env, model, episodes=num_episodes, d_r=d_r, d_h=d_h,
            time_interval=0.05, state_dim=state_dim, use_goal=use_goal)
        average_rewards.append(np.mean(returns))
        sem_values.append(sem(returns))
        success_rates.append(np.mean([r > 0 for r in returns]))
        best_distances.append(np.mean(distances))

    plot_average_rewards(average_rewards, sem_values, d_r_options,
                         title="Average Reward vs. d_r", save_path="antmaze_average_rewards_plot.png",
                         max_y=max(d_r_options) * 1.1)
    print("success rates: ", success_rates)
