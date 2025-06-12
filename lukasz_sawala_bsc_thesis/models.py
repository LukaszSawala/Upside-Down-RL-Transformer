import torch.nn as nn
import torch
import numpy as np


class NeuralNet(nn.Module):
    """
    Class defining the Neural Network used in the research as a baseline and as part of
    (multiple) policies. It is a simple feedforward neural network with
    multiple fully connected layers and ReLU activation functions.
    The output is squeezed with tanh to enforce the action range between (-1, 1).
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
    Class defining the Neural Network (larger) used on top of the BERT. It is an extension of the
    NeuralNet class with more layers.
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


class NeuralNetResNorm(nn.Module):
    """
    Class defining the Neural Network with residual connections and layer normalization.
    It is a more complex version of the NeuralNet class, with multiple layers and residual connections.
    The output is squeezed with tanh to enforce the action range between (-1, 1).
    This class is used in the AntMaze environment and is a part of the policies.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 16) -> None:
        super(NeuralNetResNorm, self).__init__()
        self.hidden_size = hidden_size
        self.act = nn.ReLU()

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)

        for _ in range(num_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.norms.append(nn.LayerNorm(hidden_size))

        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.input_norm(x)
        x = self.act(x)

        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            residual = x
            x = layer(x)
            x = norm(x)
            x = self.act(x)
            if i % 2 == 1:  # Add residual every 2 layers
                x = x + residual

        x = self.output_layer(x)
        x = torch.tanh(x)
        return x


class ActionHead(nn.Module):
    """
    This class defines the action head used in the Ant environment, specifically by the UDRLt architectures.
    It takes the hidden state from the model combined with d_r and d_h to produce the final action.
    """
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


class LargeActionHead(nn.Module):
    """
    This class defines a larger action head used in the Ant environment, specifically by the UDRLt architectures.
    It takes the hidden state from the model combined with d_r and d_h to produce the final action.
    It is a more complex version of the ActionHead class, with more layers and ReLU activations.
    """
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
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, act_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class ScalarEncoder(nn.Module):
    """
    This class defines a simple scalar encoder that takes a single scalar input
    and encodes it into a higher-dimensional space.
    It is used to encode the reward and horizon vectors (d_r, d_h) in the AntMaze environment.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )

    def forward(self, x):
        return self.net(x)


class AntMazeActionHead(nn.Module):
    """
    This class defines the action head used in the AntMaze environment, put on top of the pretrained models.
    It takes the goal location as an input, which is concatenated with the output of the pretrained model to
    produce the final action.
    """
    def __init__(self, hidden_size: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(act_dim + 2, hidden_size),  # 2 for x, y of the goal
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
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


class BertAntMazeActionHead(nn.Module):
    """
    This class defines the action head used in the AntMaze environment, put on top of the pretrained BERT model.
    It is a more complex version of the AntMazeActionHead class, with layer normalization and residual connections.
    It takes the goal location as an input, which is concatenated with the output of the pretrained model to
    produce the final action.
    """
    def __init__(self, hidden_size: int, act_dim: int, num_layers: int = 10, input_size: int = 10) -> None:
        super(BertAntMazeActionHead, self).__init__()
        self.hidden_size = hidden_size
        self.act = nn.ReLU()

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)

        for _ in range(num_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.norms.append(nn.LayerNorm(hidden_size))

        self.output_layer = nn.Linear(hidden_size, act_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.input_norm(x)
        x = self.act(x)

        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            residual = x
            x = layer(x)
            x = norm(x)
            x = self.act(x)
            if i % 2 == 1:  # Add residual every 2 layers
                x = x + residual

        x = self.output_layer(x)
        x = torch.tanh(x)
        return x


class AntNNPretrainedMazePolicy(nn.Module):
    """
    This class is a wrapper used to navigate the AntMaze environment using the NeuralNet in two ways:
    -  with the goal location (pretrained on Ant, finetuned on AntMaze)
    -  without the goal location (only pretrained on Ant).
    """
    def __init__(self, base_model, action_dim, adjusted_head=None):
        super().__init__()
        self.base_model = base_model
        if adjusted_head is not None:
            self.adjusted_head = adjusted_head
        else:
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


class AntBERTPretrainedMazePolicy(nn.Module):
    """
    This class is a wrapper used to navigate the AntMaze environment using the UDRLt-MLP in two ways:
    -  with the goal location (pretrained on Ant, finetuned on AntMaze)
    -  without the goal location (only pretrained on Ant).
    """

    def __init__(self, model_bert, state_encoder, mlp, action_dim=8, init_head=True, adjusted_head=None, hidden_size=64):
        super().__init__()
        self.state_encoder = state_encoder
        self.mlp = mlp  # the main mlp
        self.model_bert = model_bert
        if init_head:
            self.adjusted_head = AntMazeActionHead(hidden_size=hidden_size, act_dim=action_dim)
        else:
            self.adjusted_head = adjusted_head

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
    This class is a wrapper for the UDRLt-MLP model fully trained on AntMaze.
    """
    def __init__(self, model_bert, state_encoder, mlp):
        super().__init__()
        self.state_encoder = state_encoder
        self.mlp = mlp
        self.model_bert = model_bert

    def forward(self, obs, dr, dh, goal_vector, DEVICE, **kwargs):
        # convert to tensors if needed
        if not torch.is_tensor(obs):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            dr_tensor = torch.tensor([dr], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            dh_tensor = torch.tensor([dh], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            goal_tensor = torch.tensor(goal_vector, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        else:
            obs_tensor = obs.to(DEVICE)
            dr_tensor = dr.to(DEVICE)
            dh_tensor = dh.to(DEVICE)
            goal_tensor = goal_vector.to(DEVICE)

        s_encoded = self.state_encoder(obs_tensor).unsqueeze(1)
        bert_out = self.model_bert(inputs_embeds=s_encoded).last_hidden_state[:, 0]
        mlp_input = torch.cat([bert_out, dr_tensor, dh_tensor, goal_tensor], dim=1)
        return self.mlp(mlp_input)


class AntMazeNNPretrainedMazeWrapper(nn.Module):
    """
    This class is a wrapper for the NeuralNet model fully trained on AntMaze.
    """
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, obs, dr, dh, goal_vector, DEVICE, **kwargs):
        # convert to tensors if needed
        if not torch.is_tensor(obs):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            dr_tensor = torch.tensor([dr], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            dh_tensor = torch.tensor([dh], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            goal_tensor = torch.tensor(goal_vector, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        else:
            obs_tensor = obs.to(DEVICE)
            dr_tensor = dr.to(DEVICE)
            dh_tensor = dh.to(DEVICE)
            goal_tensor = goal_vector.to(DEVICE)
        mlp_input = torch.cat([obs_tensor, dr_tensor, dh_tensor, goal_tensor], dim=1)
        return self.mlp(mlp_input)
