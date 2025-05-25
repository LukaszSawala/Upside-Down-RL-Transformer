import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
import time
from models import AntMazePolicy
import gymnasium_robotics
from utils import parse_arguments

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_goal_direction(obs: dict) -> np.ndarray:
    goal_x, goal_y = obs['desired_goal']
    current_x, current_y = obs['achieved_goal']
    return np.array([goal_x - current_x, goal_y - current_y])


class PPOAgent:
    def __init__(self, model, lr=3e-4, gamma=0.99, clip_epsilon=0.2, epochs=10, batch_size=64):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size

    def select_action(self, obs_input, goal_tensor):
        # Assuming model outputs mean actions; we sample with Gaussian noise for exploration
        mean_action = self.model(obs_input, goal_tensor)
        dist = Normal(mean_action, torch.ones_like(mean_action) * 0.1)  # fixed small stddev
        action = dist.sample()
        action_log_prob = dist.log_prob(action).sum(dim=-1)
        return action.detach(), action_log_prob.detach()

    def compute_returns(self, rewards, masks, gamma):
        returns = []
        R = 0
        for r, mask in zip(reversed(rewards), reversed(masks)):
            R = r + gamma * R * mask
            returns.insert(0, R)
        return returns

    def ppo_update(self, trajectories):
        obs = torch.cat(trajectories['obs'])
        goals = torch.cat(trajectories['goals'])
        actions = torch.cat(trajectories['actions'])
        old_log_probs = torch.cat(trajectories['log_probs'])
        returns = torch.tensor(trajectories['returns'], dtype=torch.float32).to(DEVICE)

        for _ in range(self.epochs):
            # Forward pass
            mean_actions = self.model(obs, goals)
            dist = Normal(mean_actions, torch.ones_like(mean_actions) * 0.1)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

            ratios = torch.exp(log_probs - old_log_probs)
            advantages = returns - returns.mean()  # simple advantage estimate

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


def train(
    env_name="AntMaze_MediumDense-v5",
    model_type="NeuralNet",
    episodes=1000,
    max_timesteps=1000,
    lr=3e-4,
    gamma=0.99,
    clip_epsilon=0.2,
    epochs=10,
    batch_size=64,
    save_path="ppo_antmaze_model.pth"
):
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
    env = gym.make(env_name)

    from model_evaluation import load_nn_model_for_eval, load_bert_mlp_model_for_eval
    from model_evaluation import NN_MODEL_PATH, BERT_MLP_MODEL_PATH

    if model_type == "NeuralNet":
        base_model = load_nn_model_for_eval(107, 256, 8, NN_MODEL_PATH, DEVICE)
    elif model_type == "BERT_MLP":
        base_model = load_bert_mlp_model_for_eval(BERT_MLP_MODEL_PATH, DEVICE)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model = AntMazePolicy(base_model, action_dim=8).to(DEVICE)
    agent = PPOAgent(model, lr=lr, gamma=gamma, clip_epsilon=clip_epsilon, epochs=epochs, batch_size=batch_size)

    print("Starting training...")
    all_episode_rewards = []

    for ep in range(episodes):
        obs = env.reset()[0]
        ep_rewards = []
        ep_masks = []
        ep_obs = []
        ep_goals = []
        ep_actions = []
        ep_log_probs = []

        for t in range(max_timesteps):
            goal_vec = extract_goal_direction(obs)
            obs_vec = obs['observation']

            obs_tensor = torch.tensor(np.concatenate((obs_vec, [0.0, 1000.0])), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            goal_tensor = torch.tensor(goal_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            action, log_prob = agent.select_action(obs_tensor, goal_tensor)
            action_np = action.squeeze(0).cpu().numpy()

            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            ep_obs.append(obs_tensor)
            ep_goals.append(goal_tensor)
            ep_actions.append(action)
            ep_log_probs.append(log_prob)
            ep_rewards.append(reward)
            ep_masks.append(0.0 if done else 1.0)

            obs = next_obs
            if done:
                break

        returns = agent.compute_returns(ep_rewards, ep_masks, gamma)

        trajectories = {
            'obs': ep_obs,
            'goals': ep_goals,
            'actions': ep_actions,
            'log_probs': ep_log_probs,
            'returns': returns,
        }

        agent.ppo_update(trajectories)

        ep_return = sum(ep_rewards)
        all_episode_rewards.append(ep_return)
        print(f"Episode {ep + 1}/{episodes} - Return: {ep_return:.2f}")

        if (ep + 1) % 50 == 0:
            torch.save(model.state_dict(), save_path)
            print(f"Saved model checkpoint to {save_path}")

    print("Training completed.")
    torch.save(model.state_dict(), save_path)
    print(f"Final model saved to {save_path}")


if __name__ == "__main__":
    args = parse_arguments(training=True)
    train(
        env_name=args.get("env_name", "AntMaze_MediumDense-v5"),
        model_type=args.get("model_type", "NeuralNet"),
        episodes=args.get("episodes", 1000),
    )
