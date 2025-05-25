import torch.nn as nn
import torch
import numpy as np


class NeuralNet(nn.Module):
    """
    Class defining the Neural Network used in the research as a baseline.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        """
        Initializes the Neural Network.
        """
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc5 = nn.Linear(hidden_size // 4, output_size)
        self.relu = nn.ReLU()  # ReLU activation function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward call of the network.
        Returns:
            torch.Tensor: The output of the network squeezed with
            tanh - enforcing the action range between (-1, 1)
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = torch.tanh(x)  # Enforces the action range between -1 and 1
        return x


class HugeNeuralNet(nn.Module):
    """
    Class defining the Neural Network (larger) used on top of the BERT.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        """
        Initializes the Neural Network.
        """
        super(HugeNeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc6 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc7 = nn.Linear(hidden_size // 4, output_size)
        self.relu = nn.ReLU()  # ReLU activation function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward call of the network.
        Returns:
            torch.Tensor: The output of the network squeezed with
            tanh - enforcing the action range between (-1, 1)
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.fc7(x)
        x = torch.tanh(x)  # Enforces the action range between -1 and 1
        return x


class ActionHead(nn.Module):
    def __init__(self, hidden_size: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, act_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class ScalarEncoder(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )

    def forward(self, x):
        return self.net(x)


class LargeActionHead(nn.Module):
    def __init__(self, hidden_size: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, act_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)
    


class AntMazeActionHead(nn.Module):
    def __init__(self, hidden_size: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(act_dim + 2, hidden_size), # 2 for x, y of the goal
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, act_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class AntNNPretrainedMazePolicy(nn.Module):
    """
    Class definign the policy wrapper for the AntMaze transferrablity experiment
    """
    def __init__(self, base_model, action_dim):
        super().__init__()
        self.base_model = base_model
        self.adjusted_head = AntMazeActionHead(hidden_size=64, act_dim=action_dim)

    def forward(self, obs, dr, dh, goal_vector, DEVICE, use_goal=True):
        obs_input = torch.tensor(np.concatenate((obs, [dr, dh])), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        base_output = self.base_model(obs_input)
        if not use_goal:  # not using the goal location
            return base_output
        else:
            goal_tensor = (torch.tensor(goal_vector, dtype=torch.float32).unsqueeze(0).to(DEVICE))
            x = torch.cat((base_output, goal_tensor), dim=1)
            action = self.adjusted_head(x)
            return action
        return action


class AntBERTPretrainedMazePolicy(nn.Module):
    """
    THIS GUY IS FOR THE ANT-TRAINED BERTMLP
    """

    def __init__(self, model_bert, state_encoder, mlp, action_dim=8):
        super().__init__()
        self.state_encoder = state_encoder
        self.mlp = mlp  # the main mlp
        self.model_bert = model_bert
        self.adjusted_head = AntMazeActionHead(hidden_size=64, act_dim=action_dim)

    def forward(self, obs, dr, dh, goal_vector, DEVICE, use_goal=True):
        # convert to tensors
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        dr_tensor = torch.tensor([dr], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        dh_tensor = (torch.tensor([dh], dtype=torch.float32).unsqueeze(0).to(DEVICE))

        s_encoded = self.state_encoder(obs_tensor).unsqueeze(1)
        bert_out = self.model_bert(inputs_embeds=s_encoded).last_hidden_state[:, 0]
        mlp_input = torch.cat([bert_out, dr_tensor, dh_tensor], dim=1)
        base_output = self.mlp(mlp_input)

        if not use_goal:  # not using the goal location
            return base_output
        else:
            goal_tensor = (torch.tensor(goal_vector, dtype=torch.float32).unsqueeze(0).to(DEVICE))
            x = torch.cat((base_output, goal_tensor), dim=1)
            action = self.adjusted_head(x)
            return action


class AntMazeBERTPretrainedMazeWrapper(nn.Module):
    """
    THIS GUY IS FOR THE ANTMAZETRAINED BERTMLP WITH GOAL TRAINING
    """
    def __init__(self, model_bert, state_encoder, mlp):
        super().__init__()
        self.state_encoder = state_encoder
        self.mlp = mlp
        self.model_bert = model_bert

    def forward(self, obs, dr, dh, goal_vector, DEVICE, **kwargs):
        # convert to tensors
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        dr_tensor = torch.tensor([dr], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        dh_tensor = torch.tensor([dh], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        goal_tensor = torch.tensor(goal_vector, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        s_encoded = self.state_encoder(obs_tensor).unsqueeze(1)
        bert_out = self.model_bert(inputs_embeds=s_encoded).last_hidden_state[:, 0]
        mlp_input = torch.cat([bert_out, dr_tensor, dh_tensor, goal_tensor], dim=1)
        return self.mlp(mlp_input)
