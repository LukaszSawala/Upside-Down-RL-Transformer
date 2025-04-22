import torch.nn as nn
import torch


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
