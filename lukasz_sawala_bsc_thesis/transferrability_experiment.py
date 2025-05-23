import gymnasium as gym
import torch
import numpy as np
import time
import gymnasium_robotics
from models import AntMazeActionHead
from model_evaluation import (
    load_nn_model_for_eval, load_bert_mlp_model_for_eval, 
    NN_MODEL_PATH, BERT_MLP_MODEL_PATH,
)
from utils import parse_arguments

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Wrap with AntMazeActionHead ---
class AntMazePolicy(torch.nn.Module):
    def __init__(self, base_model, action_dim):
        super().__init__()
        self.base_model = base_model
        self.adjusted_head = AntMazeActionHead(hidden_size=64, act_dim=action_dim)

    def forward(self, input_vector, goal_vector):
        with torch.no_grad():
            base_output = self.base_model(input_vector)
        #concatenate
        x = torch.cat((base_output, goal_vector), dim=1)
        action = self.adjusted_head(x)
        return action
    

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
def evaluate(model, episodes=10, max_steps=1000, time_interval=0.05):
    returns = []
    for episode in range(episodes):
        print("="*50)
        print(f"Episode: {episode}")
        obs = env.reset()[0] # extract the values from the wrapped array

        done = False
        total_reward = 0
        step = 0
        best_distance = 1000

        while not done and step < max_steps:
            goal_vec = extract_goal_direction(obs)
            distance = np.linalg.norm(goal_vec)
            if distance < best_distance:
                best_distance = distance
                print("best distance found:", best_distance)
                
            d_r = 1000.0
            d_h = 1000.0

            print(obs)
            obs = obs['observation'] # exytract the values from the wrapped array

            obs_input = torch.tensor(
                np.concatenate((obs, [d_r, d_h])), dtype=torch.float32
                ).unsqueeze(0).to(DEVICE)

            goal_tensor = torch.tensor(goal_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                action_tensor = model(obs_input, goal_tensor)
                #action_tensor = model(obs_input)
            action = action_tensor.squeeze(0).cpu().numpy()

            obs, reward, terminated, truncated, _  = env.step(action)
            done = terminated or truncated

            #env.render()
            #time.sleep(time_interval)

            total_reward += reward
            step += 1

        returns.append(total_reward)

    mean_return = np.mean(returns)
    return mean_return


if __name__ == "__main__":
    args = parse_arguments(training=False)
    gym.register_envs(gymnasium_robotics)

    # print("Available AntMaze environments:")
    # antmaze_envs = []
    # for env_spec in gym.envs.registry.values():
    #     if "AntMaze" in env_spec.id:
    #         antmaze_envs.append(env_spec.id)

    # # Sort them for better readability
    # for env_id in sorted(antmaze_envs):
    #     print(f"- {env_id}")
    
    env = gym.make("AntMaze_MediumDense-v5") # DO WITH RENDER MODE HUMAN AT HOME


    if args["model_type"] == "NeuralNet":
        nn_base = load_nn_model_for_eval(107, 256, 8, NN_MODEL_PATH, DEVICE)
        #evaluate(nn_base)
        model = AntMazePolicy(nn_base, action_dim=8).to(DEVICE)
    elif args["model_type"] == "BERT_MLP":
        bert_base = load_bert_mlp_model_for_eval(BERT_MLP_MODEL_PATH, DEVICE)
        model = AntMazePolicy(bert_base, action_dim=8).to(DEVICE)
    else:
        raise ValueError(f"Unsupported model_type: {args['model_type']}")


    print(f"\nEvaluating {args['model_type']} + AntMazeActionHead:")
    result = evaluate(model)
    print(f"Average Return: {result}")
    
