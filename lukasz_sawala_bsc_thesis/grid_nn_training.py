# Dataset testing
import itertools

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from zeus.monitor import ZeusMonitor

from models import NeuralNet

INPUT_SIZE = 105 + 2  # s_t + d_r and d_t
HIDDEN_SIZE = 256
OUTPUT_SIZE = 8
CONCATENATED_DATA_PATH = "../data/processed/concatenated_data.hdf5"
BEST_MODEL_PATH = "../models/best_nn_grid.pth"

# ========= BEST MODEL FOUND==================
# {'batch_size': 16, 'learning_rate': 0.0001}
# MSE: test: 0.02467
# ============================================


# ======================================= FILE EXPLANATION ======================================

# This script is designed to train a neural network model on concatenated data from Ant-v5.
# It performs a grid search over hyperparameters such as batch size and learning rate.
# The script loads the data from an HDF5 file, splits it into training, validation, and test sets,
# and trains a neural network model using the specified hyperparameters.

# ================================================================================================

def load_data(data_path: str = CONCATENATED_DATA_PATH) -> tuple:
    """
    Load data from an HDF5 file.
    Parameters:
        data_path (str): The path to the HDF5 file.
    Returns:
        tuple: A tuple containing the input features and target labels.
    """
    with h5py.File(data_path, "r") as f:
        data = f["concatenated_data"]
        actions = data["actions"][:]
        observations = data["observations"][:]
        rewards_to_go = data["rewards_to_go"][:]
        time_to_go = data["time_to_go"][:]

    # Reshape rewards_to_go and time_to_go to be 2D with shape (1000, 1)
    rewards_to_go = rewards_to_go.reshape(-1, 1)
    time_to_go = time_to_go.reshape(-1, 1)

    # Combine the inputs into one array
    X = np.concatenate((observations, rewards_to_go, time_to_go), axis=-1)
    y = actions

    return X, y


def train_test_val_split(
    X, y, test_size=0.1, val_size=0.1, shuffle=True, random_state=42
) -> tuple:
    """
    Split the data into train, test, and validation sets.
    Parameters:
        X (numpy.ndarray): The input features.
        y (numpy.ndarray): The target labels.
        test_size (float): The proportion of the data to include in the test set.
        val_size (float): The proportion of the data to include in the validation set.
        shuffle (bool): Whether to shuffle the data before splitting.
        random_state (int): The seed used by the random number generator.
    Returns:
        tuple: A tuple containing the train, test, and validation sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=shuffle, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size / (1 - test_size),
        shuffle=shuffle,
        random_state=random_state,
    )
    print(
        f"Train-test split created: {len(X_train)} train samples, {len(X_test)} test samples, {len(X_val)} validation samples."
    )
    return X_train, X_test, X_val, y_train, y_test, y_val


def create_tensor_dataloaders(
    X_train, X_test, X_val, y_train, y_test, y_val, batch_size=32, shuffle=True
) -> tuple:
    """
    Create PyTorch DataLoader objects for the train, test, and validation sets.
    Parameters:
        X_train (numpy.ndarray): The train input features.
        X_test (numpy.ndarray): The test input features.
        X_val (numpy.ndarray): The validation input features.
        y_train (numpy.ndarray): The train target labels.
        y_test (numpy.ndarray): The test target labels.
        y_val (numpy.ndarray): The validation target labels.
    Returns:
        tuple: A tuple containing the train, test, and validation DataLoader objects.
    """
    # Convert the numpy data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, generator=torch.Generator().manual_seed(42))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, generator=torch.Generator().manual_seed(42))

    return train_loader, val_loader, test_loader


def grid_search_train(
    X_train, X_test, X_val, y_train, y_test, y_val, epochs=20, patience=2
):
    """ "
    Initialize training parameters and models"
    """
    loss_fn = nn.MSELoss()
    val_losses = []
    train_losses = []
    smallest_test_loss = float("inf")
    batch_sizes = [16, 32, 64]  # Different batch sizes to try
    learning_rates = [0.0001, 0.001]  # Different learning rates to try

    best_model = None
    best_hyperparams = None

    for batch_size, lr in itertools.product(batch_sizes, learning_rates):
        print("=======================")
        print(f"Training with batch size {batch_size} and learning rate {lr}")

        # Create data loaders with current batch size
        train_loader, val_loader, test_loader = create_tensor_dataloaders(
            X_train,
            X_test,
            X_val,
            y_train,
            y_test,
            y_val,
            batch_size=batch_size,
            shuffle=True,
        )

        # Initialize model and optimizer
        model = NeuralNet(
            input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE
        )
        optimizer = optim.Adam(model.parameters(), lr=lr)

        val_losses = []
        train_losses = []
        smallest_val_loss = float("inf")

        # TRAINING LOOP ===============================================
        for epoch in range(epochs):
            # TRAINING
            model.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # EVALUATION
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            # Early stopping
            if val_loss < smallest_val_loss:
                patience = 2
                smallest_val_loss = val_loss
                best_model_state = model.state_dict()
            else:
                patience -= 1
                if patience == 0:
                    break

            print(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
            )
        # ===============================================================

        # TESTING
        model.load_state_dict(best_model_state)
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                test_loss += loss.item()
        test_loss /= len(test_loader)

        # Track best model based on test loss
        if test_loss < smallest_test_loss:
            smallest_test_loss = test_loss
            best_model = best_model_state
            best_hyperparams = {"batch_size": batch_size, "learning_rate": lr}
            print(f"Best Model Found: {best_hyperparams} with Test Loss: {test_loss}")
            torch.save(best_model, BEST_MODEL_PATH)


if __name__ == "__main__":
    # Load the data
    X, y = load_data()

    # Split the data
    X_train, X_test, X_val, y_train, y_test, y_val = train_test_val_split(X, y)

    # monitor = ZeusMonitor(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # monitor.begin_window("grid-search")
    grid_search_train(
        X_train, X_test, X_val, y_train, y_test, y_val, epochs=20, patience=2
    )
    # mes = monitor.end_window("grid-search")
    # print(f"Training grid search took {mes.time} s and consumed {mes.total_energy} J.")
