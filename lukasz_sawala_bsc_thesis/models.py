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

class NeuralNet10(nn.Module):
    """
    Class defining the Neural Network aaa.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        """
        Initializes the Neural Network.
        """
        super(NeuralNet10, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size*2)
        self.fc3 = nn.Linear(hidden_size*2, hidden_size*2)
        self.fc4 = nn.Linear(hidden_size*2, hidden_size*2)
        self.fc5 = nn.Linear(hidden_size*2, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.fc7 = nn.Linear(hidden_size, hidden_size)
        self.fc8 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc9 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc10 = nn.Linear(hidden_size // 4, output_size)
        self.relu = nn.ReLU()  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward call of the network.
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
        x = self.relu(x)
        x = self.fc8(x)
        x = self.relu(x)
        x = self.fc9(x)
        x = self.relu(x)
        x = self.fc10(x)
        x = torch.tanh(x)  # Enforces the action range between -1 and 1
        return x

class NeuralNet12(nn.Module):
    """
    Class defining the Neural Network used in the research as a baseline.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        """
        Initializes the Neural Network.
        """
        super(NeuralNet12, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size*2)
        self.fc3 = nn.Linear(hidden_size*2, hidden_size*2)
        self.fc4 = nn.Linear(hidden_size*2, hidden_size*2)
        self.fc5 = nn.Linear(hidden_size*2, hidden_size*2)
        self.fc6 = nn.Linear(hidden_size*2, hidden_size)
        self.fc7 = nn.Linear(hidden_size, hidden_size)
        self.fc8 = nn.Linear(hidden_size, hidden_size)
        self.fc9 = nn.Linear(hidden_size, hidden_size)
        self.fc10 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc11 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc12 = nn.Linear(hidden_size // 4, output_size)
        self.relu = nn.ReLU()  

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
        x = self.relu(x)
        x = self.fc8(x)
        x = self.relu(x)
        x = self.fc9(x)
        x = self.relu(x)
        x = self.fc10(x)
        x = self.relu(x)
        x = self.fc11(x)
        x = self.relu(x)
        x = self.fc12(x)
        x = torch.tanh(x)  # Enforces the action range between -1 and 1
        return x
    

class NeuralNet16(nn.Module):
    """
    Class defining the Neural Network used in the research as a baseline.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        """
        Initializes the Neural Network.
        """
        super(NeuralNet16, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size*2)
        self.fc3 = nn.Linear(hidden_size*2, hidden_size*2)
        self.fc4 = nn.Linear(hidden_size*2, hidden_size*2)
        self.fc5 = nn.Linear(hidden_size*2, hidden_size*2)
        self.fc6 = nn.Linear(hidden_size*2, hidden_size*2)
        self.fc7 = nn.Linear(hidden_size*2, hidden_size*2)
        self.fc8 = nn.Linear(hidden_size*2, hidden_size)
        self.fc9 = nn.Linear(hidden_size, hidden_size)
        self.fc10 = nn.Linear(hidden_size, hidden_size)
        self.fc11 = nn.Linear(hidden_size, hidden_size)
        self.fc12 = nn.Linear(hidden_size, hidden_size)
        self.fc13 = nn.Linear(hidden_size, hidden_size)
        self.fc14 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc15 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc16 = nn.Linear(hidden_size // 4, output_size)
        self.relu = nn.ReLU()  

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
        x = self.relu(x)
        x = self.fc8(x)
        x = self.relu(x)
        x = self.fc9(x)
        x = self.relu(x)
        x = self.fc10(x)
        x = self.relu(x)
        x = self.fc11(x)
        x = self.relu(x)
        x = self.fc12(x)
        x = self.relu(x)
        x = self.fc13(x)
        x = self.relu(x)
        x = self.fc14(x)
        x = self.relu(x)
        x = self.fc15(x)
        x = self.relu(x)
        x = self.fc16(x)
        x = torch.tanh(x)  # Enforces the action range between -1 and 1
        return x

class NeuralNet18(nn.Module):
    """
    Class defining the Neural Network used in the research as a baseline.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        """
        Initializes the Neural Network.
        """
        super(NeuralNet18, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size*2)
        self.fc3 = nn.Linear(hidden_size*2, hidden_size*2)
        self.fc4 = nn.Linear(hidden_size*2, hidden_size*2)
        self.fc5 = nn.Linear(hidden_size*2, hidden_size*2)
        self.fc6 = nn.Linear(hidden_size*2, hidden_size*2)
        self.fc7 = nn.Linear(hidden_size*2, hidden_size*2)
        self.fc8 = nn.Linear(hidden_size*2, hidden_size*2)
        self.fc9 = nn.Linear(hidden_size*2, hidden_size)
        self.fc10 = nn.Linear(hidden_size, hidden_size)
        self.fc11 = nn.Linear(hidden_size, hidden_size)
        self.fc12 = nn.Linear(hidden_size, hidden_size)
        self.fc13 = nn.Linear(hidden_size, hidden_size)
        self.fc14 = nn.Linear(hidden_size, hidden_size)
        self.fc15 = nn.Linear(hidden_size, hidden_size)
        self.fc16 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc17 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc18 = nn.Linear(hidden_size // 4, output_size)
        self.relu = nn.ReLU()  

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
        x = self.relu(x)
        x = self.fc8(x)
        x = self.relu(x)
        x = self.fc9(x)
        x = self.relu(x)
        x = self.fc10(x)
        x = self.relu(x)
        x = self.fc11(x)
        x = self.relu(x)
        x = self.fc12(x)
        x = self.relu(x)
        x = self.fc13(x)
        x = self.relu(x)
        x = self.fc14(x)
        x = self.relu(x)
        x = self.fc15(x)
        x = self.relu(x)
        x = self.fc16(x)
        x = self.relu(x)
        x = self.fc17(x)
        x = self.relu(x)
        x = self.fc18(x)
        x = torch.tanh(x)  # Enforces the action range between -1 and 1
        return x
    
class NeuralNet20(nn.Module):
    """
    Class defining the Neural Network used in the research as a baseline.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        """
        Initializes the Neural Network.
        """
        super(NeuralNet20, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size*2)
        self.fc3 = nn.Linear(hidden_size*2, hidden_size*2)
        self.fc4 = nn.Linear(hidden_size*2, hidden_size*2)
        self.fc5 = nn.Linear(hidden_size*2, hidden_size*2)
        self.fc6 = nn.Linear(hidden_size*2, hidden_size*2)
        self.fc7 = nn.Linear(hidden_size*2, hidden_size*2)
        self.fc8 = nn.Linear(hidden_size*2, hidden_size)
        self.fc9 = nn.Linear(hidden_size, hidden_size)
        self.fc10 = nn.Linear(hidden_size, hidden_size)
        self.fc11 = nn.Linear(hidden_size, hidden_size)
        self.fc12 = nn.Linear(hidden_size, hidden_size)
        self.fc13 = nn.Linear(hidden_size, hidden_size)
        self.fc14 = nn.Linear(hidden_size, hidden_size)
        self.fc15 = nn.Linear(hidden_size, hidden_size)
        self.fc16 = nn.Linear(hidden_size, hidden_size)
        self.fc17 = nn.Linear(hidden_size, hidden_size)
        self.fc18 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc19 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc20 = nn.Linear(hidden_size // 4, output_size)
        self.relu = nn.ReLU()  

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
        x = self.relu(x)
        x = self.fc8(x)
        x = self.relu(x)
        x = self.fc9(x)
        x = self.relu(x)
        x = self.fc10(x)
        x = self.relu(x)
        x = self.fc11(x)
        x = self.relu(x)
        x = self.fc12(x)
        x = self.relu(x)
        x = self.fc13(x)
        x = self.relu(x)
        x = self.fc14(x)
        x = self.relu(x)
        x = self.fc15(x)
        x = self.relu(x)
        x = self.fc16(x)
        x = self.relu(x)
        x = self.fc17(x)
        x = self.relu(x)
        x = self.fc18(x)
        x = self.relu(x)
        x = self.fc19(x)
        x = self.relu(x)
        x = self.fc20(x)
        x = torch.tanh(x)  # Enforces the action range between -1 and 1
        return x

class NeuralNetResNorm(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 16) -> None:
        super(NeuralNet, self).__init__()
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
    This class defines the action head used in the Ant environment, specifically from BERT.
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
    


class OldAntMazeActionHead(nn.Module):
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


class LessOldAntMazeActionHead(nn.Module):
    def __init__(self, hidden_size: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(act_dim + 2, hidden_size), # 2 for x, y of the goal
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
    

class NewestAntMazeActionHead(nn.Module):
    """
    This class defines the action head used in the AntMaze environment, put on top of the pretrained models.
    It takes the goal location as an input, which is concatenated with the output of the pretrained model to
    produce the final action.
    """
    def __init__(self, hidden_size: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(act_dim + 2, hidden_size), # 2 for x, y of the goal
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
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
    

class AntMazeActionHead(nn.Module):
    def __init__(self, hidden_size: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(act_dim + 2, hidden_size), # 2 for x, y of the goal
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
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

    def __init__(self, model_bert, state_encoder, mlp, action_dim=8, init_head=True, adjusted_head = None, hidden_size=64):
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
        # convert to tensors
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        dr_tensor = torch.tensor([dr], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        dh_tensor = torch.tensor([dh], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        goal_tensor = torch.tensor(goal_vector, dtype=torch.float32).unsqueeze(0).to(DEVICE)

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
        # convert to tensors
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        dr_tensor = torch.tensor([dr], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        dh_tensor = torch.tensor([dh], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        goal_tensor = torch.tensor(goal_vector, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        mlp_input = torch.cat([obs_tensor, dr_tensor, dh_tensor, goal_tensor], dim=1)
        return self.mlp(mlp_input)