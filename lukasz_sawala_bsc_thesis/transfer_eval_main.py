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
    AntMazeBERTPretrainedMazeWrapper, AntMazeNNPretrainedMazeWrapper,
    HugeNeuralNet, NeuralNetResNorm
)
from model_evaluation import (
    load_nn_model_for_eval, load_bert_mlp_model_for_eval,
    NN_MODEL_PATH, BERT_MLP_MODEL_PATH,
)
from utils import parse_arguments
from model_evaluation import plot_average_rewards, print_available_antmaze_envs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ANTMAZE_BERT_PATH = "../models/antmaze_tiny-18_512.pth" # condition 3
ANTMAZE_NN_PATH = "../models/antmaze_NN-18_512.pth" # condition 3

# ANTMAZE_BERT_PATH = "antmazeMERGEDinit_tiny-18_512"  # condition 4
# ANTMAZE_NN_PATH = "antmazeMERGEDinit_NN-18_512"  # condition 4



def load_antmaze_nn_model_for_eval(checkpoint_path: str, device: str, initialize_from_scratch: bool = False) -> NeuralNetResNorm:
    """
    Loads the AntMaze NN model components for evaluation.
    Args:
        checkpoint_path (str): The path to the checkpoint file.
        device (str): The device to load the model on.
    Returns:
        nn_base (NeuralNet): The loaded model.
    """
    nn_base = NeuralNetResNorm(input_size=31, hidden_size=512, output_size=8, num_layers=18).to(device)
    if initialize_from_scratch:
        return nn_base
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    nn_base.load_state_dict(checkpoint["nn"])

    # Set models to evaluation mode
    nn_base.eval()

    return nn_base


def load_antmaze_bertmlp_model_for_eval(checkpoint_path: str, device: str, initialize_from_scratch: bool = False) -> tuple:
    """
    Loads the AntMaze BERT MLP model components for evaluation.
    Returns:
        model_bert (AutoModel): The loaded BERT model.
        state_encoder (nn.Linear): The loaded state encoder.
        mlp (NeuralNet): The loaded MLP model.
    """
    config = AutoConfig.from_pretrained("prajjwal1/bert-tiny")
    config.vocab_size = 1
    config.max_position_embeddings = 1

    # Initialize components
    model_bert = AutoModel.from_config(config).to(device)
    state_encoder = nn.Linear(27, config.hidden_size).to(device)

    # hidden size + 4 for d_r, d_h and x y values of the goal vector
    mlp = NeuralNetResNorm(input_size=config.hidden_size + 4, hidden_size=512, output_size=8, num_layers=18).to(device)
    
    if initialize_from_scratch:
        return model_bert, state_encoder, mlp
    
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
    """
    Evaluate a single given model in the AntMaze environment.

    Parameters:
        env (gym.Env): The AntMaze environment.
        model: The model to evaluate (wrapped to accept multiple input variables).
        episodes (int): The number of episodes to run for each condition.
        time_interval (float): Time to sleep between steps when rendering the environment.
        d_r (float): The desired reward.
        d_h (float): The desired horizon.
        state_dim (int): The number of dimensions in the state vector.
        use_goal (bool): Whether to use the goal direction in the model.

    Returns:
        obtained_returns (list): A list of the total rewards obtained in each episode.
        best_distances (list): A list of the minimum distances to the goal achieved in each episode.
    """
    best_distances = []
    obtained_returns = []
    for episode in range(episodes):
        print(f"Episode: {episode}")
        obs = env.reset()[0]  # extract the values from the wrapped array
        # print(obs)
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

            # env.render()    # uncomment this if you want to see the environment
            # time.sleep(time_interval)
        obtained_returns.append(total_reward)
        if best_distance < 1:
            print("goal reached!")
        best_distances.append(best_distance)
        #print(f"Episode {episode} finished with total reward: {total_reward}, best distance: {best_distance}")

    print("minimum return:", min(obtained_returns), "maximum return:", max(obtained_returns), "average return:", np.mean(obtained_returns))
    return obtained_returns, best_distances


def transfer_eval_main(args):
    gym.register_envs(gymnasium_robotics)
    # print_available_antmaze_envs() # check whether its compatible
    env = gym.make("AntMaze_MediumDense-v5")  # ALTENRATIVE: "AntMaze_Medium_Diverse_GR-v4" # render mode human to see whats up

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
        model_path = args["model_path"] if "model_path" in args.keys() else ANTMAZE_BERT_PATH
        model_components = load_antmaze_bertmlp_model_for_eval(model_path, DEVICE)
        model = AntMazeBERTPretrainedMazeWrapper(*model_components).to(DEVICE)
        state_dim = 27  # reduced state space due to dataset mismatch
        use_goal = True
    elif args["model_type"] == "ANTMAZE_NN":
        model_path = args["model_path"] if "model_path" in args.keys() else ANTMAZE_NN_PATH
        model = load_antmaze_nn_model_for_eval(model_path, DEVICE)
        model = AntMazeNNPretrainedMazeWrapper(model).to(DEVICE)
        state_dim = 27  # reduced state space due to dataset mismatch
        use_goal = True
    else:
        raise ValueError(f"Unsupported model_type: {args['model_type']}")

    d_h = 1000.0
    d_r_options = [i * 50 for i in range(args["d_r_array_length"])]  
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
            time_interval=0.005, state_dim=state_dim, use_goal=use_goal)
        average_rewards.append(np.mean(returns))
        sem_values.append(sem(returns))
        success_rates.append(np.mean([d < 1 for d in distances]))

    save_path = f"antmaze_{args['model_type']}_d_r_eval_results.png"
    
    if "return_without_plotting" in args.keys() and args["return_without_plotting"]:
        return average_rewards, sem_values, success_rates

    plot_average_rewards(average_rewards, sem_values, d_r_options,
                         title="Average Reward vs. d_r", save_path=save_path,
                         max_y=max(d_r_options) * 1.1)
    print("success rates: ", success_rates, "average:", np.mean(success_rates))

if __name__ == "__main__":
    args = parse_arguments(training=False)
    transfer_eval_main(args)
