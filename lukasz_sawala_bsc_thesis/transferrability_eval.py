import gymnasium as gym
import torch
import numpy as np
import time
import gymnasium_robotics
from scipy.stats import sem
from models import AntMazePolicy
from model_evaluation import (
    load_nn_model_for_eval, load_bert_mlp_model_for_eval, 
    NN_MODEL_PATH, BERT_MLP_MODEL_PATH,
)
from utils import parse_arguments
from model_evaluation import plot_average_rewards, print_available_antmaze_envs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# --- Evaluation Loop ---
def evaluate(model, episodes=10, time_interval=0.05, d_r = 5.0, d_h = 1000.0):
    mean_returns = []
    best_distances = []
    for episode in range(episodes):
        print(f"Episode: {episode}")
        obs = env.reset()[0] # extract the values from the wrapped array
        done = False
        rewards = []
        d_h_copy, d_r_copy = d_h, d_r
        best_distance = 1000
        while not done and d_h_copy > 0:
            goal_vec = extract_goal_direction(obs)
            distance = np.linalg.norm(goal_vec)
            if distance < best_distance:
                best_distance = distance

            obs = obs['observation'] # exytract the values from the wrapped array
            obs_input = torch.tensor(
                np.concatenate((obs, [d_r_copy, d_h_copy])), dtype=torch.float32
                ).unsqueeze(0).to(DEVICE)

            goal_tensor = torch.tensor(goal_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                action_tensor = model(obs_input, goal_tensor)
                #action_tensor = model(obs_input)  # if not using the goal location
            action = action_tensor.squeeze(0).cpu().numpy()

            obs, reward, terminated, truncated, _  = env.step(action)
            rewards.append(reward)
            d_r_copy -= reward
            d_h_copy -= 1
            done = terminated or truncated

            #env.render()
            #time.sleep(time_interval)
        mean_returns.append(np.mean(rewards))
        best_distances.append(best_distance)

    return mean_returns, best_distances


if __name__ == "__main__":
    args = parse_arguments(training=False)
    gym.register_envs(gymnasium_robotics)
    # print_available_antmaze_envs() # check whether its compatible
    
    env = gym.make("AntMaze_MediumDense-v5") # DO WITH RENDER MODE HUMAN AT HOME
    

    if args["model_type"] == "NeuralNet":
        nn_base = load_nn_model_for_eval(107, 256, 8, NN_MODEL_PATH, DEVICE)
        model = AntMazePolicy(nn_base, action_dim=8).to(DEVICE)
    elif args["model_type"] == "BERT_MLP":
        bert_base = load_bert_mlp_model_for_eval(BERT_MLP_MODEL_PATH, DEVICE)
        model = AntMazePolicy(bert_base, action_dim=8).to(DEVICE)
    else:
        raise ValueError(f"Unsupported model_type: {args['model_type']}")

    d_h = 1000.0
    d_r_options = [i * 0.1 for i in range(11)]
    num_episodes = args["episodes"]
    average_rewards = []
    sem_values = []
    success_rates = []
    best_distances = []
    for d_r in d_r_options:
        print("=" * 50)
        print("Trying with d_r:", d_r)
        returns, distances = evaluate(model, episodes=num_episodes, d_r=d_r, d_h=d_h, time_interval=0.05)
        average_rewards.append(np.mean(returns))
        sem_values.append(sem(returns))
        success_rates.append(np.mean([r > 0 for r in returns]))
        best_distances.append(np.mean(distances))

    plot_average_rewards(average_rewards, sem_values, d_r_options,
                         title="Average Reward vs. d_r", save_path="antmaze_average_rewards_plot.png",
                         max_y=1.0)
    print("success rates: ", success_rates)